"""Backtest a EUR/USD strategy based on the US-Germany 2-year yield spread."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    from .nlp_engine import combine_quant_and_nlp_signals, load_sentiment_cache
except ImportError:
    from nlp_engine import combine_quant_and_nlp_signals, load_sentiment_cache


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the yield-spread backtest."""

    start_date: str = "2006-01-01"
    end_date: str | None = None
    german_yield_series: str = "IRLTLT01DEM156N"
    fx_source: Literal["FRED", "YAHOO"] = "FRED"
    fx_ticker: str = "EURUSD=X"
    bund_ticker: str = "DE02Y"
    rolling_windows: tuple[int, ...] = (60, 120)
    export_window: int = 60
    entry_threshold: float = 1.0
    transaction_cost_pips: float = 1.5
    pip_size: float = 0.0001
    trading_days_per_year: int = 252
    sentiment_cache_path: Path = Path(__file__).resolve().parent / \
        "data" / "sentiment_cache.json"
    output_path: Path = Path(__file__).resolve().parent / "df_20yr_results.csv"


def _download_fred_series(
    series_id: str,
    start: str,
    end: str,
    output_name: str,
) -> pd.Series:
    """Download and parse one observation series from the public FRED CSV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(url, na_values=[".", "", "NA"])
    date_column = next(
        (column for column in frame.columns if column.lower() in
         {"observation_date", "date"}),
        None,
    )
    if date_column is None or series_id not in frame.columns:
        raise RuntimeError(
            f"Unexpected FRED response for series {series_id!r}.")
    series = pd.to_numeric(frame.set_index(date_column)
                           [series_id], errors="coerce")
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series.loc[start:end].dropna().sort_index()
    if series.empty:
        raise RuntimeError(
            f"No FRED observations for {series_id!r} between {start} and {end}."
        )
    return series.rename(output_name)


def _download_yahoo_fx(ticker: str, start: str, end: str) -> pd.Series:
    """Download the EUR/USD spot series from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install yfinance to use fx_source='YAHOO'.") from exc
    history = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if history.empty or "Close" not in history:
        raise RuntimeError(f"No Yahoo Finance data returned for {ticker!r}.")
    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.astype(float).rename("EURUSD")


def load_market_data(config: BacktestConfig) -> pd.DataFrame:
    """Download EUR/USD and yields, then reconcile US/European holidays.

    FRED's IRLTLT01DEM156N is a German long-term (10-year) rate, not a
    2-year yield. It is used here only as the explicitly requested proxy.
    """
    end = config.end_date or date.today().isoformat()
    us02y = _download_fred_series("DGS2", config.start_date, end, "US02Y")
    german_proxy = _download_fred_series(
        config.german_yield_series,
        config.start_date,
        end,
        "DE02Y_Proxy",
    )
    if config.fx_source == "FRED":
        eurusd = _download_fred_series(
            "DEXUSEU", config.start_date, end, "EURUSD")
    else:
        eurusd = _download_yahoo_fx(config.fx_ticker, config.start_date, end)

    data = pd.concat(
        [eurusd.rename("EURUSD"), us02y, german_proxy],
        axis=1,
    )

    business_days = pd.date_range(config.start_date, end, freq="B")
    source_and_business_dates = data.index.union(business_days).sort_values()
    data = (
        data.reindex(source_and_business_dates)
        .ffill()
        .reindex(business_days)
        .dropna()
    )
    data.index.name = "Date"

    if data.empty:
        raise RuntimeError("No overlapping market data was available.")
    if (data["EURUSD"] <= 0).any():
        raise ValueError("EUR/USD prices must be positive.")

    return data


def _calculate_trade_returns(
    positions: pd.Series,
    asset_returns: pd.Series,
    prices: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, list[float]]:
    """Calculate net daily returns and net returns for each position episode."""
    daily_returns: list[float] = []
    trade_returns: list[float] = []
    active_trade: int | None = None
    previous_position = 0.0

    for current_position, asset_return, price in zip(
        positions.to_numpy(),
        asset_returns.to_numpy(),
        prices.to_numpy(),
    ):
        cost_per_unit = config.transaction_cost_pips * config.pip_size / price
        daily_return = current_position * asset_return

        if current_position != previous_position:
            if previous_position != 0 and active_trade is not None:
                daily_return -= cost_per_unit
                trade_returns[active_trade] -= cost_per_unit
                active_trade = None

            if current_position != 0:
                daily_return -= cost_per_unit
                trade_returns.append(0.0)
                active_trade = len(trade_returns) - 1

        if current_position != 0 and active_trade is not None:
            trade_returns[active_trade] += current_position * asset_return

        daily_returns.append(daily_return)
        previous_position = current_position

    return pd.Series(daily_returns, index=positions.index), trade_returns


def _max_drawdown_duration(equity: pd.Series) -> int:
    """Return the longest underwater period in trading days."""
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity.to_numpy()))
    )[1:]
    drawdown = equity.to_numpy() / running_peak - 1.0

    longest = current = 0
    for is_underwater in drawdown < 0:
        current = current + 1 if is_underwater else 0
        longest = max(longest, current)
    return longest


def calculate_metrics(
    returns: pd.Series,
    trade_returns: list[float] | None,
    trading_days_per_year: int,
) -> dict[str, float | int]:
    """Calculate return, risk, drawdown, and trade statistics."""
    returns = returns.dropna()
    equity = np.exp(returns.cumsum())
    daily_std = returns.std(ddof=1)
    downside_std = returns[returns < 0].std(ddof=1)

    simple_returns = np.expm1(returns)
    gross_gains = simple_returns[simple_returns > 0].sum()
    gross_losses = -simple_returns[simple_returns < 0].sum()

    trade_returns = trade_returns or []
    return {
        "Cumulative Return (%)": (equity.iloc[-1] - 1) * 100,
        "Annualized Return (%)": (
            np.expm1(returns.sum() * trading_days_per_year / len(returns)) * 100
        ),
        "Annualized Volatility (%)": (
            daily_std * np.sqrt(trading_days_per_year) * 100
            if pd.notna(daily_std)
            else np.nan
        ),
        "Sharpe Ratio": (
            returns.mean() / daily_std * np.sqrt(trading_days_per_year)
            if pd.notna(daily_std) and daily_std != 0
            else np.nan
        ),
        "Sortino Ratio": (
            returns.mean() / downside_std * np.sqrt(trading_days_per_year)
            if pd.notna(downside_std) and downside_std != 0
            else np.nan
        ),
        "Maximum Drawdown (%)": (
            (equity / pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
             .cummax().iloc[1:].set_axis(equity.index) - 1).min()
            * 100
        ),
        "Max Drawdown Duration (trading days)": _max_drawdown_duration(equity),
        "Profit Factor": (
            gross_gains / gross_losses if gross_losses > 0 else np.nan
        ),
        "Win Rate (%)": (
            np.mean(np.array(trade_returns) > 0) * 100
            if trade_returns
            else np.nan
        ),
        "Trades": len(trade_returns),
    }


def _aligned_sentiment_scores(
    trading_index: pd.DatetimeIndex,
    cache_path: Path,
) -> pd.DataFrame:
    """Align cached bank scores to trading dates using only prior-day events."""
    scores = load_sentiment_cache(cache_path)
    for column in ("FED_Score", "ECB_Score"):
        if column not in scores.columns:
            scores[column] = np.nan
    if scores.empty:
        return pd.DataFrame(
            np.nan,
            index=trading_index,
            columns=["FED_Score", "ECB_Score"],
        )

    scores.index = pd.to_datetime(scores.index).tz_localize(None).normalize()
    combined_index = scores.index.union(trading_index).sort_values()
    expanded = (
        scores[["FED_Score", "ECB_Score"]]
        .reindex(combined_index)
        .ffill()
    )
    prior_positions = expanded.index.searchsorted(
        trading_index, side="left") - 1
    aligned = pd.DataFrame(
        np.nan,
        index=trading_index,
        columns=["FED_Score", "ECB_Score"],
    )
    has_prior_event = prior_positions >= 0
    aligned.iloc[has_prior_event] = expanded.iloc[prior_positions[has_prior_event]].to_numpy()
    return aligned


def _calculate_regime_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate Sharpe, maximum drawdown, and total return for one regime."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown (%)": np.nan,
            "Total Return (%)": np.nan,
        }
    daily_std = returns.std(ddof=1)
    equity = np.exp(returns.cumsum())
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity.to_numpy()))
    )[1:]
    drawdown = equity.to_numpy() / running_peak - 1.0
    return {
        "Sharpe Ratio": (
            returns.mean() / daily_std * np.sqrt(252)
            if pd.notna(daily_std) and daily_std != 0
            else np.nan
        ),
        "Maximum Drawdown (%)": float(drawdown.min() * 100),
        "Total Return (%)": float(np.expm1(returns.sum()) * 100),
    }


def build_regime_report(
    results: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Compare benchmark, quant-only, and combined returns across four eras."""
    window = config.export_window
    regimes = (
        ("2006-2010 GFC and emergency cuts", "2006-01-01", "2010-12-31"),
        ("2011-2015 European debt crisis and zero rates", "2011-01-01", "2015-12-31"),
        ("2016-2020 ZIRP/NIRP", "2016-01-01", "2020-12-31"),
        ("2021-2026 inflation shock and tightening", "2021-01-01", "2026-12-31"),
    )
    strategy_columns = {
        "Buy & Hold": "Benchmark_Return",
        f"Quant {window}D": f"Strategy_Return_Quant_{window}",
        f"Combined {window}D": f"Strategy_Return_Combined_{window}",
    }
    rows: list[dict[str, str | float]] = []
    backtest_end = pd.Timestamp(config.end_date or date.today())
    for regime, start, scheduled_end in regimes:
        period_end = min(pd.Timestamp(scheduled_end), backtest_end)
        period = results.loc[start:period_end]
        for strategy, column in strategy_columns.items():
            if column not in period:
                continue
            rows.append(
                {
                    "Regime": regime,
                    "Strategy": strategy,
                    **_calculate_regime_metrics(period[column]),
                }
            )
    return pd.DataFrame(rows).set_index(["Regime", "Strategy"])


def run_backtest(
    config: BacktestConfig = BacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 2006-to-current backtest and export its dashboard-ready data.

    The combined strategy is flat until both banks have cached sentiment
    scores. Sentiment events become usable on the next business day.
    """
    if not config.rolling_windows:
        raise ValueError("At least one rolling window is required.")
    if config.export_window not in config.rolling_windows:
        raise ValueError("export_window must be included in rolling_windows.")

    data = load_market_data(config)
    results = data.copy()
    results["Spread"] = results["US02Y"] - results["DE02Y_Proxy"]
    results["Asset_Return"] = np.log(
        results["EURUSD"] / results["EURUSD"].shift(1)
    )
    results["Benchmark_Return"] = results["Asset_Return"]
    sentiment = _aligned_sentiment_scores(
        pd.DatetimeIndex(results.index), Path(config.sentiment_cache_path)
    )
    results["NLP_FED_Score"] = sentiment["FED_Score"]
    results["NLP_ECB_Score"] = sentiment["ECB_Score"]
    results["NLP_Sentiment_Diff"] = (
        results["NLP_FED_Score"] - results["NLP_ECB_Score"]
    )

    metric_rows: dict[str, dict[str, float | int]] = {
        "Buy & Hold": calculate_metrics(
            results["Benchmark_Return"].dropna(),
            None,
            config.trading_days_per_year,
        )
    }

    for window in config.rolling_windows:
        rolling_mean = results["Spread"].rolling(window).mean()
        rolling_std = results["Spread"].rolling(window).std(ddof=1)
        z_score = (results["Spread"] - rolling_mean) / rolling_std
        results[f"ZScore_{window}"] = z_score

        # Shift once: the position used for day t is based on information at t-1.
        raw_signal = pd.Series(
            np.select(
                [
                    z_score > config.entry_threshold,
                    z_score < -config.entry_threshold,
                ],
                [-1, 1],
                default=0,
            ),
            index=results.index,
            dtype=float,
        )
        position = raw_signal.shift(1).fillna(0.0)
        results[f"Signal_{window}"] = position

        quant_return, quant_trade_returns = _calculate_trade_returns(
            position,
            results["Asset_Return"].fillna(0.0),
            results["EURUSD"],
            config,
        )
        results[f"Strategy_Return_Quant_{window}"] = quant_return
        results[f"Equity_Quant_{window}"] = np.exp(quant_return.cumsum())

        combined_signal = pd.Series(0.0, index=results.index)
        for row_index, (quant, fed, ecb) in enumerate(
            zip(position, sentiment["FED_Score"], sentiment["ECB_Score"])
        ):
            if pd.notna(fed) and pd.notna(ecb):
                combined_signal.iloc[row_index] = combine_quant_and_nlp_signals(
                    int(quant), float(fed), float(ecb)
                )
        results[f"Signal_Combined_{window}"] = combined_signal
        combined_return, combined_trade_returns = _calculate_trade_returns(
            combined_signal,
            results["Asset_Return"].fillna(0.0),
            results["EURUSD"],
            config,
        )
        results[f"Strategy_Return_Combined_{window}"] = combined_return
        results[f"Equity_Combined_{window}"] = np.exp(combined_return.cumsum())

        metric_rows[f"Quant Strategy {window}D"] = calculate_metrics(
            quant_return,
            quant_trade_returns,
            config.trading_days_per_year,
        )
        metric_rows[f"Combined Strategy {window}D"] = calculate_metrics(
            combined_return,
            combined_trade_returns,
            config.trading_days_per_year,
        )

    results["Benchmark_Equity"] = np.exp(
        results["Benchmark_Return"].fillna(0.0).cumsum()
    )
    export_window = config.export_window
    results["ZScore"] = results[f"ZScore_{export_window}"]
    results["Signal_Quant"] = results[f"Signal_{export_window}"]
    results["Signal_Combined"] = results[f"Signal_Combined_{export_window}"]
    results["Equity_Strategy"] = results[f"Equity_Combined_{export_window}"]

    export_columns = [
        "EURUSD",
        "US02Y",
        "DE02Y_Proxy",
        "Spread",
        "Benchmark_Return",
        "ZScore",
        "NLP_FED_Score",
        "NLP_ECB_Score",
        "NLP_Sentiment_Diff",
        "Signal_Quant",
        "Signal_Combined",
        "Equity_Strategy",
        "Benchmark_Equity",
    ]
    for window in config.rolling_windows:
        export_columns.extend(
            [
                f"ZScore_{window}",
                f"Signal_{window}",
                f"Signal_Combined_{window}",
                f"Strategy_Return_Quant_{window}",
                f"Strategy_Return_Combined_{window}",
                f"Equity_Quant_{window}",
                f"Equity_Combined_{window}",
            ]
        )
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results[export_columns].to_csv(output_path, index_label="Date")

    report = pd.DataFrame.from_dict(metric_rows, orient="index")
    report.index.name = "Strategy"
    return results.dropna(subset=["Asset_Return"]), report


if __name__ == "__main__":
    config = BacktestConfig()
    try:
        df_results, performance_report = run_backtest(config)
    except Exception as error:
        raise SystemExit(f"Backtest failed: {error}") from error

    print("\nPerformance Report")
    print(performance_report.round(3).to_string())
    regime_report = build_regime_report(df_results, config)
    print("\nMacro Regime Stability Report")
    print(regime_report.round(3).to_string())
    print(f"\nCSV exported to: {Path(config.output_path).resolve()}")
    print("German yield input: FRED IRLTLT01DEM156N (long-term proxy, not 2-year)")
    if df_results["NLP_Sentiment_Diff"].notna().sum() == 0:
        print(
            "NLP cache has no complete FED/ECB score pairs; combined strategy stayed flat.")
    print(f"\nResults rows: {len(df_results):,}")
    print(f"Date range: {df_results.index.min().date()} to "
          f"{df_results.index.max().date()}")
    print("\nDataFrame available as: df_results")
