# eurusd-monetary-policy-trading


A systematic quantitative macro trading strategy and intelligence dashboard that trades the EUR/USD currency pair based on 2-year interest rate yield differentials (US02Y vs DE02Y) combined with Natural Language Processing (NLP) on Federal Reserve (FOMC) and European Central Bank (ECB) policy statements.

---

## Executive Summary

Exchange rate movements in major liquid pairs like EUR/USD are primarily driven by relative monetary policy divergence. When central banks diverge in policy trajectory (e.g., the Federal Reserve maintaining higher policy rates while the European Central Bank cuts aggressively), capital flows toward the currency offering higher relative yield.

This project implements this macroeconomic framework into an automated system:

1. Quantitative Signal Engine: Measures the velocity and direction of the US-German 2-Year Yield Spread using a rolling Z-Score model to capture macroeconomic momentum.
2. AI / NLP Intelligence Engine: Employs Large Language Models (LLMs) to analyze FOMC and ECB policy statements and assign a structured Hawk/Dove score (-5.0 to +5.0) to filter false breakouts.
3. Interactive Dashboard: Built with Streamlit and Plotly to visualize equity curves, regime breakdowns, Z-score entry signals, and NLP sentiment metrics.

---

## Backtest Performance (2006 - 2026)

The strategy was backtested across 20 years of daily historical data (5,418 trading sessions) spanning four distinct macroeconomic regimes: the 2008 Financial Crisis, the European Sovereign Debt Crisis, the Zero/Negative Interest Rate Era, and the Post-COVID Inflationary Cycle.

| Metric | 60-Day Lookback | 120-Day Lookback | Benchmark (EUR/USD Buy & Hold) |
| :--- | :--- | :--- | :--- |
| Total Cumulative Return | +61.9% | +104.4% | -2.1% |
| Annualized Sharpe Ratio | 0.37 | 0.52 | N/A |
| Total Trading Sessions | 5,418 | 5,418 | 5,418 |
| Primary Data Sources | FRED API / Yahoo Finance | FRED API / Yahoo Finance | Yahoo Finance |

Key Finding: Extending the Z-score lookback window from 60 to 120 days increased the Sharpe Ratio from 0.37 to 0.52 and doubled overall strategy performance. This confirms that central bank policy divergence acts as a slow-moving, persistent macroeconomic trend.

---

## Project Structure

```text
macro-systematic-fx-trading/
├── data/
│   ├── df_20yr_results.csv        # 20-year historical backtest dataset
│   └── sentiment_cache.json       # Cached LLM Hawk/Dove scores for Fed/ECB
├── quant_engine.py                # Data ingestion, yield spread & Z-Score engine
├── nlp_engine.py                  # LLM Sentiment Analysis module
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
