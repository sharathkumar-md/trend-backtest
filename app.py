import logging
import logging.config
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Logging — configure once at startup (visible in terminal running streamlit)
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "level":     "DEBUG",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "strategy": {"level": "DEBUG", "propagate": True},
        "data":     {"level": "DEBUG", "propagate": True},
        "metrics":  {"level": "DEBUG", "propagate": True},
    },
})

logger = logging.getLogger(__name__)

from config import (
    ALL_ASSET_CLASSES, NAME_TO_SYMBOL, PORTFOLIO_SIZE,
    RISK_FACTOR, TICKERS,
)
from data import fetch_ohlcv
from metrics import build_portfolio_equity, calculate_metrics
from strategy import calculate_indicators, get_current_signal, run_backtest

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trend Following Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trend Following Dashboard")
st.caption(
    "Clenow-style CTA · 50-day breakout · 3-ATR trailing stop · "
    "Mark-to-Market equity · $1M portfolio"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    asset_class = st.selectbox("Asset Class", ALL_ASSET_CLASSES)
    name_list   = (
        [n for t in TICKERS.values() for n in t]
        if asset_class == "All"
        else list(TICKERS[asset_class].keys())
    )
    ticker_name = st.selectbox("Ticker", name_list)
    symbol      = NAME_TO_SYMBOL[ticker_name]

    st.divider()
    portfolio_value = st.number_input(
        "Portfolio Size ($)", min_value=10_000,
        value=PORTFOLIO_SIZE, step=10_000, format="%d",
    )
    risk_factor_input = st.number_input(
        "Risk Factor (per ATR)", min_value=0.0001, max_value=0.01,
        value=RISK_FACTOR, format="%.4f",
        help="Clenow default: 0.002 (20 bps). Fraction of portfolio risked per ATR move.",
    )

    st.divider()
    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "**Strategy Rules**\n"
        "- Filter: DMA50 > DMA100 (long) / DMA50 < DMA100 (short)\n"
        "- Entry: 50-day high/low breakout (next-bar open)\n"
        "- Exit: 3-ATR trailing stop from peak/trough\n"
        "- Size: (Portfolio × Risk Factor) / ATR"
    )

# ---------------------------------------------------------------------------
# Flat ticker list for tabs 2 & 3
# ---------------------------------------------------------------------------
ticker_items = [
    (name, sym, cls)
    for cls, tickers in TICKERS.items()
    for name, sym in tickers.items()
]

# ---------------------------------------------------------------------------
# Signal badge styles
# ---------------------------------------------------------------------------
SIGNAL_STYLE = {
    "LONG":  ("🟢", "#00d4aa"),
    "SHORT": ("🔴", "#ff6b6b"),
    "FLAT":  ("⬜", "#888888"),
}

# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------
_HIGHER_BETTER = {
    "Total Return": True, "CAGR": True, "Sharpe Ratio": True,
    "Sortino Ratio": True, "Calmar Ratio": True,
    "Win Rate": True, "Profit Factor": True, "Avg Win ($)": True,
    "Max Drawdown": False, "Avg Loss ($)": False,
}

def _metric_delta(key: str, raw: float) -> tuple[str | None, str]:
    """Return (delta_str, delta_color) for st.metric."""
    if key not in _HIGHER_BETTER:
        return None, "off"
    if math.isinf(raw) or math.isnan(raw):
        return None, "off"
    higher = _HIGHER_BETTER[key]
    good   = (higher and raw >= 0) or (not higher and raw >= -0.15)
    label  = "▲ Good" if good else "▼ Caution"
    colour = "normal" if good else "inverse"
    return label, colour


# ---------------------------------------------------------------------------
# Charting helpers
# ---------------------------------------------------------------------------

def _marker_prices(df: pd.DataFrame, dates: pd.Index, col: str = "Open") -> list:
    """Look up prices at trade dates; None if date not in df."""
    return [float(df.loc[d, col]) if d in df.index else None for d in dates]


def make_backtest_chart(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    name: str,
) -> go.Figure:
    """
    3-panel Plotly chart:
      Row 1 — MTM Equity Curve vs Buy & Hold
      Row 2 — Price + DMA50 + DMA100 + trade markers
      Row 3 — Drawdown (%)
    """
    bh_norm  = df["Close"] / df["Close"].iloc[0] * equity.iloc[0]
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100  # as %

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.40, 0.22],
        vertical_spacing=0.03,
        subplot_titles=[
            "MTM Equity Curve vs Buy & Hold",
            f"{name} — Price, DMA50, DMA100",
            "Drawdown (%)",
        ],
    )

    # ── Row 1: Equity ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Strategy (MTM)", line=dict(color="#00d4aa", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bh_norm.index, y=bh_norm.values,
        name="Buy & Hold", line=dict(color="#aaaaaa", width=1.5, dash="dot"),
    ), row=1, col=1)

    # ── Row 2: Price + DMAs ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#e8e8e8", width=1.5),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["DMA_50"],
        name="DMA 50", line=dict(color="#ffa07a", width=1.2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["DMA_100"],
        name="DMA 100", line=dict(color="#87ceeb", width=1.2),
    ), row=2, col=1)

    # ── Trade markers (use Open price — actual execution price) ──────────────
    if not trades.empty and "Entry Date" in trades.columns:
        longs  = trades[trades["Direction"] == "LONG"]
        shorts = trades[trades["Direction"] == "SHORT"]

        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs["Entry Date"],
                y=_marker_prices(df, longs["Entry Date"], "Open"),
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", color="#00d4aa", size=11,
                            line=dict(width=1, color="#ffffff")),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=longs["Exit Date"],
                y=_marker_prices(df, longs["Exit Date"], "Open"),
                mode="markers", name="Long Exit",
                marker=dict(symbol="x-thin", color="#ff6b6b", size=11,
                            line=dict(width=2, color="#ff6b6b")),
            ), row=2, col=1)

        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts["Entry Date"],
                y=_marker_prices(df, shorts["Entry Date"], "Open"),
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", color="#ff6b6b", size=11,
                            line=dict(width=1, color="#ffffff")),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=shorts["Exit Date"],
                y=_marker_prices(df, shorts["Exit Date"], "Open"),
                mode="markers", name="Short Exit",
                marker=dict(symbol="x-thin", color="#ffa07a", size=11,
                            line=dict(width=2, color="#ffa07a")),
            ), row=2, col=1)

    # ── Row 3: Drawdown ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", name="Drawdown %",
        line=dict(color="#ff6b6b", width=1),
        fillcolor="rgba(255,107,107,0.20)",
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=65, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price",         row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %",    row=3, col=1)
    return fig


def make_portfolio_chart(portfolio_eq: pd.Series) -> go.Figure:
    roll_max = portfolio_eq.cummax()
    dd       = (portfolio_eq - roll_max) / roll_max * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=["Combined Portfolio Equity", "Drawdown (%)"],
    )
    fig.add_trace(go.Scatter(
        x=portfolio_eq.index, y=portfolio_eq.values,
        fill="tozeroy", name="Portfolio",
        line=dict(color="#00d4aa", width=2),
        fillcolor="rgba(0,212,170,0.12)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", name="Drawdown %",
        line=dict(color="#ff6b6b", width=1),
        fillcolor="rgba(255,107,107,0.18)",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=500,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %",   row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Metric card grid
# ---------------------------------------------------------------------------
_METRIC_ROWS = [
    ["Total Return", "CAGR", "Max Drawdown", "Sharpe Ratio"],
    ["Sortino Ratio", "Calmar Ratio", "Win Rate", "Profit Factor"],
    ["Avg Win ($)", "Avg Loss ($)", "Time in Market", "Total Trades"],
]

def render_metrics(perf: dict):
    for row_keys in _METRIC_ROWS:
        cols = st.columns(4)
        for col, key in zip(cols, row_keys):
            if key not in perf:
                continue
            m      = perf[key]
            delta, dcolour = _metric_delta(key, m["raw"])
            col.metric(label=key, value=m["fmt"], delta=delta,
                       delta_color=dcolour)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Single Ticker Backtest",
    "📡 Current Signals",
    "🗂 Portfolio Overview",
])

# ===========================================================================
# TAB 1 — Single Ticker Backtest
# ===========================================================================
with tab1:
    if not run_btn:
        st.info("Select a ticker in the sidebar and click **▶ Run Backtest** to begin.")
    else:
        logger.info(
            "Backtest triggered: %s (%s) | portfolio=$%.0f | risk=%.4f",
            ticker_name, symbol, portfolio_value, risk_factor_input,
        )
        with st.spinner(f"Fetching {ticker_name} ({symbol}) …"):
            raw_df = fetch_ohlcv(symbol)

        if raw_df.empty:
            st.error(
                f"Could not fetch sufficient data for **{ticker_name}** ({symbol}). "
                "Try a different ticker."
            )
        else:
            with st.spinner("Running backtest …"):
                df     = calculate_indicators(raw_df)
                equity, trades, in_pos = run_backtest(df, portfolio_value, risk_factor_input)
                perf   = calculate_metrics(equity, trades, in_pos)
                siginfo = get_current_signal(df)

            # ── Signal banner ─────────────────────────────────────────────────
            sig          = siginfo["signal"]
            emoji, colour = SIGNAL_STYLE[sig]
            filter_txt   = (
                "DMA50 > DMA100 ✅" if siginfo["long_filter"] else
                "DMA50 < DMA100 ✅" if siginfo["short_filter"] else
                "DMA50 ≈ DMA100 ❌"
            )
            st.markdown(
                f"**Signal (next open):** {emoji} "
                f"<span style='color:{colour};font-size:1.2rem;font-weight:700'>{sig}</span>"
                f" &nbsp;|&nbsp; Close: **{siginfo['price']:,}**"
                f" &nbsp;|&nbsp; ATR: **{siginfo['atr']:,}**"
                f" &nbsp;|&nbsp; {filter_txt}",
                unsafe_allow_html=True,
            )

            # ── Chart ─────────────────────────────────────────────────────────
            st.plotly_chart(
                make_backtest_chart(df, equity, trades, ticker_name),
                use_container_width=True,
            )

            # ── Metrics ───────────────────────────────────────────────────────
            st.subheader("Performance Metrics")
            render_metrics(perf)

            # ── Trade log ─────────────────────────────────────────────────────
            if not trades.empty:
                with st.expander(
                    f"Trade Log — {len(trades)} trades "
                    f"(incl. force-close if any)", expanded=False
                ):
                    disp = trades.copy()
                    disp["PnL ($)"] = disp["PnL ($)"].map(lambda x: f"${x:,.2f}")
                    # Colour rows by P/L
                    def _colour_pnl(row):
                        try:
                            val = float(row["PnL ($)"].replace("$", "").replace(",", ""))
                        except Exception:
                            val = 0
                        colour = (
                            "rgba(0,212,170,0.12)" if val > 0 else
                            "rgba(255,107,107,0.12)" if val < 0 else ""
                        )
                        return [f"background-color:{colour}"] * len(row)

                    st.dataframe(
                        disp.style.apply(_colour_pnl, axis=1),
                        use_container_width=True,
                    )
            else:
                st.info("No completed trades in the 2-year window for this ticker.")

# ===========================================================================
# TAB 2 — Live Signal Scanner
# ===========================================================================
with tab2:
    st.subheader("📡 Live Signal Scanner — All Tickers")
    st.caption("Signals based on last available daily close. Cached 1 hour.")

    rows     = []
    progress = st.progress(0, text="Scanning tickers …")
    logger.info("Signal scan: %d tickers", len(ticker_items))

    for i, (name, sym, cls) in enumerate(ticker_items):
        raw = fetch_ohlcv(sym)
        if not raw.empty:
            try:
                df_s  = calculate_indicators(raw)
                info  = get_current_signal(df_s)
                rows.append({
                    "Ticker":       name,
                    "Class":        cls,
                    "Signal":       info["signal"],
                    "Price":        info["price"],
                    "ATR":          info["atr"],
                    "DMA 50":       info["dma_50"],
                    "DMA 100":      info["dma_100"],
                    "50D High":     info["high_50"],
                    "50D Low":      info["low_50"],
                    "Long Filter":  "✅" if info["long_filter"]  else "❌",
                    "Short Filter": "✅" if info["short_filter"] else "❌",
                })
            except Exception as exc:
                logger.warning("Signal scan failed for %s: %s", name, exc)
        progress.progress((i + 1) / len(ticker_items), text=f"Scanning {name} …")

    progress.empty()

    if rows:
        sig_df = pd.DataFrame(rows)

        longs  = (sig_df["Signal"] == "LONG").sum()
        shorts = (sig_df["Signal"] == "SHORT").sum()
        flats  = (sig_df["Signal"] == "FLAT").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 LONG",  longs)
        c2.metric("🔴 SHORT", shorts)
        c3.metric("⬜ FLAT",  flats)

        def _hl_signal(row):
            if row["Signal"] == "LONG":
                return ["background-color:rgba(0,212,170,0.15)"] * len(row)
            if row["Signal"] == "SHORT":
                return ["background-color:rgba(255,107,107,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sig_df.style.apply(_hl_signal, axis=1),
            use_container_width=True, height=520,
        )
    else:
        st.warning("No signal data could be fetched. Check your internet connection.")

# ===========================================================================
# TAB 3 — Portfolio Overview
# ===========================================================================
with tab3:
    st.subheader("🗂 Portfolio Overview — All Tickers Combined")
    st.caption(
        "Backtests every ticker independently, combines daily MTM PnL "
        "into one equity curve. Calendar gaps filled with 0 (flat)."
    )

    if st.button("🔄 Build Portfolio", key="build_portfolio"):
        ticker_equities: dict = {}
        ticker_perf:     list = []

        prog2 = st.progress(0, text="Backtesting …")
        for i, (name, sym, cls) in enumerate(ticker_items):
            raw = fetch_ohlcv(sym)
            if not raw.empty:
                try:
                    df_p           = calculate_indicators(raw)
                    eq, tr, ip     = run_backtest(df_p, portfolio_value, risk_factor_input)
                    ticker_equities[name] = eq
                    m = calculate_metrics(eq, tr, ip)
                    ticker_perf.append({
                        "Ticker":        name,
                        "Class":         cls,
                        "Total Return":  m["Total Return"]["fmt"],
                        "CAGR":          m["CAGR"]["fmt"],
                        "Max Drawdown":  m["Max Drawdown"]["fmt"],
                        "Sharpe":        m["Sharpe Ratio"]["fmt"],
                        "Sortino":       m["Sortino Ratio"]["fmt"],
                        "Calmar":        m["Calmar Ratio"]["fmt"],
                        "Win Rate":      m["Win Rate"]["fmt"],
                        "Profit Factor": m["Profit Factor"]["fmt"],
                        "Trades":        m["Total Trades"]["fmt"],
                        "Time in Mkt":   m["Time in Market"]["fmt"],
                    })
                except Exception as exc:
                    logger.warning("Portfolio backtest failed for %s: %s", name, exc)
            prog2.progress((i + 1) / len(ticker_items), text=f"Backtesting {name} …")

        prog2.empty()

        if ticker_equities:
            portfolio_eq = build_portfolio_equity(ticker_equities)
            st.plotly_chart(make_portfolio_chart(portfolio_eq), use_container_width=True)

            # Portfolio-level metrics
            st.subheader("Portfolio-Level Metrics")
            dummy_trades  = pd.DataFrame()
            dummy_in_pos  = pd.Series(False, index=portfolio_eq.index)
            port_perf     = calculate_metrics(portfolio_eq, dummy_trades, dummy_in_pos)
            render_metrics(port_perf)

            st.subheader(f"Per-Ticker Results ({len(ticker_perf)} tickers)")
            perf_df = pd.DataFrame(ticker_perf)

            def _hl_return(val):
                try:
                    v = float(str(val).replace("%", "").replace("+", ""))
                    if v > 0:
                        return "color: #00d4aa"
                    if v < 0:
                        return "color: #ff6b6b"
                except Exception:
                    pass
                return ""

            st.dataframe(
                perf_df.style.map(_hl_return, subset=["Total Return", "CAGR"]),
                use_container_width=True,
            )
        else:
            st.warning("No ticker data could be fetched. Check internet connection.")
    else:
        st.info("Click **🔄 Build Portfolio** to backtest all tickers and see the combined equity curve.")