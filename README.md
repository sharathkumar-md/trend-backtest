# Trend Following Backtest Dashboard

A Clenow-style CTA (Commodity Trading Advisor) trend-following strategy backtester and live signal dashboard built with Streamlit.

**Live App:** https://trend-backtest.streamlit.app/
**GitHub:** https://github.com/sharathkumar-md/trend-backtest

---

## Strategy Rules

| Rule | Detail |
|---|---|
| **Trend Filter** | Long only if DMA-50 > DMA-100 · Short only if DMA-50 < DMA-100 |
| **Entry** | 50-day High breakout (Long) · 50-day Low breakdown (Short) |
| **Execution** | Signal on day T close → trade at day T+1 open |
| **Exit** | 3-ATR trailing stop from position peak/trough |
| **Position Size** | `(Portfolio × Risk Factor) / ATR` — Clenow formula |
| **Portfolio** | $1,000,000 · Risk Factor: 0.002 (20 bps) |

---

## Asset Universe (24 tickers via yfinance)

| Class | Tickers |
|---|---|
| Equity | Nifty 50, S&P 500, FTSE 100, Nikkei 225, Hang Seng |
| Currency | EUR/USD, GBP/USD, USD/JPY, USD/INR, USD/CHF, USD/CNH |
| Commodity | Crude Oil, Natural Gas, Gold, Silver, Copper, Soybean, Corn, Sugar, Coffee, Lumber |
| Bonds | US 10Y |
| Crypto | Bitcoin, Ethereum |

---

## Dashboard Tabs

### Tab 1 — Single Ticker Backtest
- 3-panel Plotly chart: MTM equity curve vs Buy & Hold · Price + DMA50/100 + trade markers · Drawdown %
- 12 performance metrics: Total Return, CAGR, Max Drawdown, Sharpe, Sortino, Calmar, Win Rate, Profit Factor, Avg Win/Loss, Time in Market, Total Trades
- Colour-coded trade log (green = win, red = loss)

### Tab 2 — Live Signal Scanner
- Scans all 24 tickers and shows current signal (LONG / SHORT / FLAT)
- Displays: Price, ATR, DMA50, DMA100, 50D High/Low, trend filter status

### Tab 3 — Portfolio Overview
- Combines all ticker equity curves into one portfolio equity curve
- Portfolio-level metrics + per-ticker performance table

---

## Technical Implementation

- **ATR:** True Wilder's method — SMA seed for first N bars, then `(ATR_prev × (N-1) + TR) / N`
- **Equity:** Mark-to-market every bar (unrealised PnL included) — not just realised cash
- **No lookahead bias:** `High.shift(1).rolling(50)` ensures today's bar excluded from breakout level
- **Open positions:** Force-closed at final bar's close and recorded in trade log
- **Calendar gaps:** Filled with 0 PnL in portfolio aggregation

---

## Run Locally

```bash
git clone https://github.com/sharathkumar-md/trend-backtest.git
cd trend-backtest
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## Performance Metrics Explained

| Metric | Description |
|---|---|
| Total Return | (End Value / Start Value) - 1 |
| CAGR | Compound Annual Growth Rate |
| Max Drawdown | Peak-to-trough decline — the "pain" metric |
| Sharpe Ratio | (Return - RiskFree) / Volatility, annualised |
| Sortino Ratio | Sharpe using only downside deviation |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Win Rate | Winning trades / (Winning + Losing trades) |
| Profit Factor | Gross Profit / Gross Loss — target > 1.2 |
| Time in Market | % of days with an open position |

> Reference: *Following the Trend* by Andreas F. Clenow
