"""Streamlit dashboard for the macro quant and central-bank NLP strategy."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "df_20yr_results.csv"
CACHE_PATH = BASE_DIR / "data" / "sentiment_cache.json"
ROLLING_WINDOWS = (60, 120)
REGIMES = {
    "Full history": (date(2006, 1, 1), date(2026, 12, 31)),
    "2006-2010 | GFC & emergency cuts": (date(2006, 1, 1), date(2010, 12, 31)),
    "2011-2015 | European debt crisis": (date(2011, 1, 1), date(2015, 12, 31)),
    "2016-2020 | ZIRP / NIRP": (date(2016, 1, 1), date(2020, 12, 31)),
    "2021-2026 | Inflation & tightening": (date(2021, 1, 1), date(2026, 12, 31)),
    "Custom range": (None, None),
}


st.set_page_config(
    page_title="Macro Intelligence & Quant Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading the historical backtest...")
def load_results(csv_path: str, modified_ns: int) -> pd.DataFrame:
    """Read and validate the exported backtest, cached by file modification."""
    del modified_ns  # Included in the cache key to invalidate after a new export.
    frame = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.sort_index()
    required = {
        "EURUSD",
        "Spread",
        "Benchmark_Return",
        "Benchmark_Equity",
        "ZScore_60",
        "ZScore_120",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "The backtest export is missing required columns: "
            + ", ".join(missing)
            + ". Run quant_engine.py to regenerate it."
        )
    return frame


@st.cache_data(show_spinner=False)
def load_sentiment_documents(cache_path: str, modified_ns: int) -> list[dict[str, Any]]:
    """Read cached NLP analyses and their source text for the NLP tab."""
    del modified_ns
    path = Path(cache_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("analyses", {})
    documents: list[dict[str, Any]] = []
    for cache_key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        analysis = entry.get("analysis")
        if not isinstance(analysis, dict):
            continue
        try:
            score = float(analysis["hawk_dove_score"])
            statement_date = pd.to_datetime(analysis["statement_date"])
            bank = str(analysis["central_bank"])
        except (KeyError, TypeError, ValueError):
            continue
        if bank not in {"FED", "ECB"} or not -5 <= score <= 5:
            continue
        documents.append(
            {
                "Date": statement_date,
                "Central bank": bank,
                "Hawk / Dove": score,
                "Inflation stance": analysis.get("inflation_stance", "Unknown"),
                "Growth stance": analysis.get("growth_stance", "Unknown"),
                "Key drivers": " · ".join(analysis.get("key_drivers", [])),
                "Trader summary": analysis.get("summary_for_trader", ""),
                "Source text": entry.get("source_text", ""),
                "Cache key": cache_key,
            }
        )
    return sorted(documents, key=lambda row: row["Date"], reverse=True)


def _strategy_series(
    frame: pd.DataFrame,
    window: int,
    use_ai_filter: bool,
) -> tuple[pd.Series, pd.Series, str]:
    """Select daily returns and position series with export compatibility."""
    mode = "Combined" if use_ai_filter else "Quant"
    return_column = f"Strategy_Return_{mode}_{window}"
    if return_column not in frame:
        if use_ai_filter and f"Strategy_Return_Combined_{window}" not in frame:
            fallback = "Signal_Combined" if "Signal_Combined" in frame else "Signal_Quant"
            returns = frame["Benchmark_Return"].fillna(
                0) * frame[fallback].shift(1).fillna(0)
        else:
            legacy_equity = "Equity_Strategy" if use_ai_filter else f"Equity_Quant_{window}"
            if legacy_equity not in frame:
                raise ValueError(
                    f"No {window}-day {mode.lower()} series in the CSV. "
                    "Regenerate it by running quant_engine.py."
                )
            returns = np.log(frame[legacy_equity] /
                             frame[legacy_equity].shift(1))
    else:
        returns = frame[return_column]

    signal_column = f"Signal_{mode}_{window}"
    if signal_column not in frame:
        signal_column = f"Signal_{window}" if f"Signal_{window}" in frame else (
            "Signal_Combined" if use_ai_filter else "Signal_Quant"
        )
    signal = frame[signal_column].fillna(0.0)
    return returns.fillna(0.0), signal, mode


def _trade_statistics(
    daily_returns: pd.Series,
    positions: pd.Series,
) -> tuple[int, float]:
    """Estimate completed position-episode count and win rate."""
    trade_log_returns: list[float] = []
    active_log_return: float | None = None
    previous_position = 0.0
    for daily_return, current_position in zip(daily_returns, positions):
        changed = current_position != previous_position
        if changed:
            if active_log_return is not None:
                trade_log_returns.append(active_log_return)
                active_log_return = None
            if current_position != 0:
                active_log_return = float(daily_return)
        elif active_log_return is not None:
            active_log_return += float(daily_return)
        previous_position = float(current_position)
    if active_log_return is not None:
        trade_log_returns.append(active_log_return)
    if not trade_log_returns:
        return 0, float("nan")
    win_rate = float(np.mean(np.expm1(trade_log_returns) > 0) * 100)
    return len(trade_log_returns), win_rate


def _performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    positions: pd.Series,
) -> dict[str, float | int]:
    """Compute performance metrics over the currently filtered period."""
    strategy_returns = strategy_returns.fillna(0.0)
    benchmark_returns = benchmark_returns.fillna(0.0)
    equity = np.exp(strategy_returns.cumsum())
    benchmark_equity = np.exp(benchmark_returns.cumsum())
    daily_std = strategy_returns.std(ddof=1)
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity.to_numpy()))
    )[1:]
    drawdown = equity.to_numpy() / running_peak - 1.0
    simple_returns = np.expm1(strategy_returns)
    gross_wins = simple_returns[simple_returns > 0].sum()
    gross_losses = -simple_returns[simple_returns < 0].sum()
    trades, win_rate = _trade_statistics(strategy_returns, positions)
    return {
        "strategy_return": float((equity.iloc[-1] - 1) * 100),
        "benchmark_return": float((benchmark_equity.iloc[-1] - 1) * 100),
        "sharpe": (
            float(strategy_returns.mean() / daily_std * np.sqrt(252))
            if pd.notna(daily_std) and daily_std != 0
            else float("nan")
        ),
        "max_drawdown": float(drawdown.min() * 100),
        "profit_factor": (
            float(gross_wins / gross_losses) if gross_losses > 0 else float("nan")
        ),
        "win_rate": win_rate,
        "trades": trades,
    }


def _equity_curve(returns: pd.Series) -> pd.Series:
    """Normalize a filtered log-return series to a starting value of one."""
    return np.exp(returns.fillna(0.0).cumsum())


def _drawdown_curve(returns: pd.Series) -> pd.Series:
    """Build an underwater series from daily log returns."""
    equity = _equity_curve(returns)
    return equity / equity.cummax() - 1.0


def _style_chart(figure: go.Figure, height: int = 410) -> go.Figure:
    """Apply the shared restrained dark chart theme."""
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", color="#DCE7E4", size=12),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="rgba(170,190,185,0.12)", zeroline=False)
    return figure


def _sentiment_gauge(bank: str, score: float | None) -> go.Figure:
    """Create a labeled -5 to +5 hawk/dove indicator gauge."""
    if score is None:
        figure = go.Figure()
        figure.add_annotation(
            text="No cached score",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 17, "color": "#9eada7"},
        )
        figure.update_layout(
            template="plotly_dark",
            height=220,
            margin=dict(l=22, r=22, t=45, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]},
            title={"text": bank, "font": {"size": 15}},
        )
        return figure

    value = score
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "valueformat": "+.1f", "font": {"size": 34}},
            title={"text": bank, "font": {"size": 15}},
            gauge={
                "axis": {"range": [-5, 5], "tickvals": [-5, -2.5, 0, 2.5, 5]},
                "bar": {"color": "#83D6B3" if value >= 0 else "#F28B72", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [-5, -1], "color": "rgba(242,139,114,0.22)"},
                    {"range": [-1, 1], "color": "rgba(214,220,214,0.10)"},
                    {"range": [1, 5], "color": "rgba(131,214,179,0.22)"},
                ],
                "threshold": {"line": {"color": "#F4B860", "width": 3}, "thickness": 0.8, "value": 0},
            },
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=22, r=22, t=45, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7E4"),
    )
    return figure


def _render_sentiment_tab(cache_path: Path) -> None:
    """Render latest bank gauges and historical cached NLP analyses."""
    try:
        modified_ns = cache_path.stat().st_mtime_ns if cache_path.exists() else 0
        documents = load_sentiment_documents(str(cache_path), modified_ns)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read sentiment cache: {error}")
        return

    st.markdown("### Central-bank tone")
    latest_by_bank: dict[str, dict[str, Any]] = {}
    for document in documents:
        latest_by_bank.setdefault(document["Central bank"], document)
    gauge_columns = st.columns(2)
    for column, bank in zip(gauge_columns, ("FED", "ECB")):
        latest = latest_by_bank.get(bank)
        with column:
            st.plotly_chart(
                _sentiment_gauge(
                    bank, latest["Hawk / Dove"] if latest else None),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            if latest:
                st.caption(
                    f"Latest cached statement · {latest['Date'].date()} · "
                    f"{latest['Inflation stance']} inflation / {latest['Growth stance']} growth"
                )
            else:
                st.caption("No cached analysis yet")

    if not documents:
        st.info(
            "The local sentiment cache is empty. Analyze FED and ECB statements "
            "with nlp_engine.py to populate the gauges and historical table."
        )
        return

    st.markdown("### Historical statement analyses")
    table = pd.DataFrame(documents)
    table["Source preview"] = table["Source text"].map(
        lambda text: (text[:360].strip() +
                      "…") if text else "Not stored in this cache entry"
    )
    st.dataframe(
        table[
            [
                "Date",
                "Central bank",
                "Hawk / Dove",
                "Inflation stance",
                "Growth stance",
                "Key drivers",
                "Trader summary",
                "Source preview",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Hawk / Dove": st.column_config.NumberColumn(format="%+.1f"),
            "Source preview": st.column_config.TextColumn(width="large"),
            "Trader summary": st.column_config.TextColumn(width="large"),
            "Key drivers": st.column_config.TextColumn(width="large"),
        },
    )
    selected_key = st.selectbox(
        "Statement details",
        options=[document["Cache key"] for document in documents],
        format_func=lambda key: next(
            f"{document['Central bank']} · {document['Date'].date()}"
            for document in documents
            if document["Cache key"] == key
        ),
    )
    selected = next(
        document for document in documents if document["Cache key"] == selected_key)
    with st.expander("Read full cached statement and analysis"):
        if selected["Source text"]:
            st.text(selected["Source text"])
        else:
            st.caption(
                "Original source text was not stored for this cache entry.")
        st.markdown(f"**Key drivers:** {selected['Key drivers']}")
        st.markdown(f"**Trader summary:** {selected['Trader summary']}")


def main() -> None:
    """Render the interactive macro strategy dashboard."""
    st.markdown(
        """
        <style>
        .stApp { background: #111715; color: #e7efec; }
        [data-testid="stSidebar"] { background: #171f1c; border-right: 1px solid #2a3732; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c5d2cc !important; }
        .block-container { padding-top: 1.35rem; padding-bottom: 2.2rem; max-width: 1560px; }
        .hero-kicker { color: #8bd4b4; font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
        .hero-title { color: #f2f5f2; font-size: 2rem; line-height: 1.12; font-weight: 720; margin: .24rem 0 .35rem; }
        .hero-subtitle { color: #9eada7; font-size: .96rem; margin-bottom: .7rem; }
        .tech-badge { display: inline-block; border: 1px solid #35443e; background: #1b2521; color: #cbd8d2; border-radius: 4px; padding: 4px 9px; margin: 0 6px 6px 0; font-size: .76rem; }
        div[data-testid="stMetric"] { background: #19221f; border: 1px solid #2a3732; border-radius: 6px; padding: 13px 15px; min-height: 108px; }
        div[data-testid="stMetricLabel"] { color: #aab9b2; }
        div[data-testid="stMetricValue"] { color: #f2f5f2; }
        .section-heading { color: #e7efec; font-size: 1.05rem; font-weight: 650; margin: .2rem 0 .65rem; }
        .data-note { color: #91a29a; font-size: .8rem; }
        hr { border-color: #2a3732; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-kicker">Systematic macro · EUR/USD</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Macro Intelligence &amp; Quant Trading Strategy</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Rate-differential signals confirmed by central-bank language.</div>',
        unsafe_allow_html=True,
    )
    badges = "".join(
        f'<span class="tech-badge">{name}</span>'
        for name in (
            "Python",
            "Streamlit",
            "Plotly",
            "YFinance / FRED",
            "OpenAI / LiteLLM",
            "Pandas",
        )
    )
    st.markdown(badges, unsafe_allow_html=True)

    if not RESULTS_PATH.exists():
        st.error(f"Backtest export not found: {RESULTS_PATH}")
        st.code("python quant_engine.py", language="bash")
        st.stop()
    try:
        modified_ns = RESULTS_PATH.stat().st_mtime_ns
        results = load_results(str(RESULTS_PATH), modified_ns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the backtest export: {error}")
        st.stop()

    min_date = results.index.min().date()
    max_date = results.index.max().date()
    with st.sidebar:
        st.markdown("## Strategy controls")
        selected_window = st.radio(
            "Rolling Z-score window",
            options=ROLLING_WINDOWS,
            format_func=lambda value: f"{value} days",
            horizontal=True,
        )
        selected_regime = st.selectbox(
            "Analysis period", options=list(REGIMES))
        preset_start, preset_end = REGIMES[selected_regime]
        if selected_regime == "Custom range":
            date_range = st.slider(
                "Date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD",
            )
        else:
            start_bound = max(min_date, preset_start)
            end_bound = min(max_date, preset_end)
            date_range = st.slider(
                "Date range",
                min_value=min_date,
                max_value=max_date,
                value=(start_bound, end_bound),
                format="YYYY-MM-DD",
                key=f"range_{selected_regime}",
            )
        use_ai_filter = st.toggle(
            "Activate AI Hawk / Dove filter", value=False)
        st.divider()
        st.caption(
            f"Export updated: {pd.Timestamp(RESULTS_PATH.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"Market data: {min_date} — {max_date}")
        st.caption(
            "German rate is a long-term yield proxy, not a 2-year Bund yield.")

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    filtered = results.loc[start_date:end_date].copy()
    if filtered.empty:
        st.warning("No observations are available for this date range.")
        st.stop()

    try:
        selected_returns, selected_signal, mode = _strategy_series(
            results, selected_window, use_ai_filter
        )
        benchmark_returns = results["Benchmark_Return"].fillna(0.0)
    except (KeyError, ValueError) as error:
        st.error(str(error))
        st.stop()
    selected_returns = selected_returns.loc[filtered.index]
    selected_signal = selected_signal.loc[filtered.index]
    benchmark_returns = benchmark_returns.loc[filtered.index]
    metrics = _performance_metrics(
        selected_returns, benchmark_returns, selected_signal)

    if use_ai_filter and filtered["NLP_Sentiment_Diff"].isna().all():
        st.info(
            "No paired FED/ECB scores are cached yet. The AI-confirmed strategy is flat for this period.")

    first_metric_row = st.columns(3)
    first_metric_row[0].metric(
        "Cumulative return",
        f"{metrics['strategy_return']:+.2f}%",
        delta=f"Benchmark {metrics['benchmark_return']:+.2f}%",
    )
    first_metric_row[1].metric("Sharpe ratio", f"{metrics['sharpe']:.2f}" if np.isfinite(
        metrics["sharpe"]) else "—")
    first_metric_row[2].metric(
        "Max drawdown", f"{metrics['max_drawdown']:.2f}%")
    second_metric_row = st.columns(3)
    second_metric_row[0].metric("Profit factor", f"{metrics['profit_factor']:.2f}" if np.isfinite(
        metrics["profit_factor"]) else "—")
    second_metric_row[1].metric("Win rate", f"{metrics['win_rate']:.1f}%" if np.isfinite(
        metrics["win_rate"]) else "—")
    second_metric_row[2].metric("Trades executed", f"{metrics['trades']:,}")

    st.markdown(
        f'<div class="data-note">Showing {start_date.date()} to {end_date.date()} · '
        f'{selected_window}-day Z-score · {"AI-confirmed" if use_ai_filter else "Quant-only"} positions</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    overview_tab, nlp_tab = st.tabs(["Strategy overview", "AI sentiment lab"])
    with overview_tab:
        st.markdown('<div class="section-heading">Equity curves</div>',
                    unsafe_allow_html=True)
        equity_figure = go.Figure()
        for window, color in ((120, "#83D6B3"), (60, "#F4B860")):
            window_returns, _, _ = _strategy_series(
                results, window, use_ai_filter)
            window_returns = window_returns.loc[filtered.index]
            equity_figure.add_trace(
                go.Scatter(
                    x=filtered.index,
                    y=_equity_curve(window_returns),
                    name=f"Strategy {window}d",
                    mode="lines",
                    line=dict(color=color, width=2),
                )
            )
        equity_figure.add_trace(
            go.Scatter(
                x=filtered.index,
                y=_equity_curve(benchmark_returns),
                name="EUR/USD buy & hold",
                mode="lines",
                line=dict(color="#a4aeb5", width=1.8, dash="dot"),
            )
        )
        equity_figure.update_layout(
            title="Growth of 1.0 · selected period", yaxis_title="Equity multiple")
        st.plotly_chart(_style_chart(equity_figure), use_container_width=True)

        left_chart, right_chart = st.columns([1.35, 1])
        with left_chart:
            st.markdown(
                '<div class="section-heading">Rate spread &amp; signal levels</div>', unsafe_allow_html=True)
            signal_column = f"Signal_Combined_{selected_window}" if use_ai_filter else f"Signal_{selected_window}"
            if signal_column not in filtered:
                signal_column = "Signal_Combined" if use_ai_filter else "Signal_Quant"
            signals = filtered[signal_column].fillna(0.0)
            z_column = f"ZScore_{selected_window}"
            z_scores = filtered[z_column] if z_column in filtered else filtered["ZScore"]
            spread_figure = make_subplots(specs=[[{"secondary_y": True}]])
            spread_figure.add_trace(
                go.Scatter(
                    x=filtered.index, y=filtered["Spread"], name="US - German yield proxy", line=dict(color="#83D6B3", width=1.7)),
                secondary_y=False,
            )
            spread_figure.add_trace(
                go.Scatter(x=filtered.index, y=z_scores,
                           name=f"Z-score {selected_window}d", line=dict(color="#F4B860", width=1.5)),
                secondary_y=True,
            )
            for threshold, color in ((1, "#F28B72"), (-1, "#83D6B3")):
                spread_figure.add_hline(
                    y=threshold,
                    line_dash="dot",
                    line_color=color,
                    opacity=0.7,
                    secondary_y=True,
                    annotation_text=f"Z = {threshold:+d}",
                    annotation_position="top left" if threshold > 0 else "bottom left",
                )
            long_entries = signals.eq(1) & signals.shift(1).fillna(0).ne(1)
            short_entries = signals.eq(-1) & signals.shift(1).fillna(0).ne(-1)
            spread_figure.add_trace(
                go.Scatter(x=filtered.index[long_entries], y=filtered.loc[long_entries, "Spread"],
                           name="Long entry", mode="markers", marker=dict(color="#83D6B3", size=8, symbol="triangle-up")),
                secondary_y=False,
            )
            spread_figure.add_trace(
                go.Scatter(x=filtered.index[short_entries], y=filtered.loc[short_entries, "Spread"],
                           name="Short entry", mode="markers", marker=dict(color="#F28B72", size=8, symbol="triangle-down")),
                secondary_y=False,
            )
            spread_figure.update_yaxes(
                title_text="Yield spread (percentage points)", secondary_y=False)
            spread_figure.update_yaxes(
                title_text="Rolling Z-score", secondary_y=True)
            spread_figure.update_layout(title="Macro rates and entries")
            st.plotly_chart(_style_chart(spread_figure, 390),
                            use_container_width=True)

        with right_chart:
            st.markdown(
                '<div class="section-heading">Underwater drawdown</div>', unsafe_allow_html=True)
            drawdown = _drawdown_curve(selected_returns) * 100
            drawdown_figure = go.Figure(
                go.Scatter(
                    x=filtered.index,
                    y=drawdown,
                    name="Drawdown",
                    mode="lines",
                    line=dict(color="#F28B72", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(242,139,114,0.19)",
                )
            )
            drawdown_figure.update_layout(
                title=f"{mode} strategy · peak-to-trough", yaxis_title="Drawdown (%)")
            st.plotly_chart(_style_chart(drawdown_figure, 390),
                            use_container_width=True)

        st.markdown("<div class='data-note'>Signals use the previous session’s information. Yield spread uses the FRED German long-term series as a proxy.</div>", unsafe_allow_html=True)

    with nlp_tab:
        _render_sentiment_tab(CACHE_PATH)


if __name__ == "__main__":
    main()
