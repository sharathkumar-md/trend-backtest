import logging
import math

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE

logger = logging.getLogger(__name__)


def calculate_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    in_position: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute standard CTA performance metrics on a mark-to-market equity curve.

    Parameters
    ----------
    equity      : daily MTM portfolio value (cash + unrealised PnL)
    trades      : closed-trade log from run_backtest
    in_position : bool series — True on days with an open position
    risk_free   : annualised risk-free rate (default 5 %)

    Returns
    -------
    dict[str, {"fmt": str, "raw": float}]
    """
    logger.info("Calculating metrics: %d bars | %d trades", len(equity), len(trades))

    start_val = equity.iloc[0]
    end_val   = equity.iloc[-1]

    # ── Total Return ──────────────────────────────────────────────────────────
    total_return = (end_val / start_val) - 1

    # ── CAGR ──────────────────────────────────────────────────────────────────
    days  = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1 / 365.25)
    cagr  = (end_val / start_val) ** (1 / years) - 1

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())   # always ≤ 0

    # ── Sharpe Ratio (annualised, excess over risk-free) ─────────────────────
    daily_returns = equity.pct_change().dropna()
    mean_daily    = daily_returns.mean()
    vol_daily     = daily_returns.std()
    rf_daily      = risk_free / 252
    sharpe = ((mean_daily - rf_daily) / vol_daily * math.sqrt(252)) if vol_daily > 0 else 0.0

    # ── Sortino Ratio (downside deviation only) ───────────────────────────────
    downside = daily_returns[daily_returns < rf_daily] - rf_daily
    downside_vol = math.sqrt((downside ** 2).mean()) * math.sqrt(252) if len(downside) > 0 else 0.0
    sortino = ((mean_daily - rf_daily) * 252 / downside_vol) if downside_vol > 0 else 0.0

    # ── Calmar Ratio (CAGR / |Max Drawdown|) ─────────────────────────────────
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("inf")

    # ── Trade statistics ─────────────────────────────────────────────────────
    total_trades  = len(trades)
    win_rate      = 0.0
    profit_factor = 1.0
    avg_win       = 0.0
    avg_loss      = 0.0

    if total_trades > 0 and "PnL ($)" in trades.columns:
        pnl    = trades["PnL ($)"]
        wins   = pnl[pnl > 0]
        losses = pnl[pnl < 0]   # breakeven (=0) excluded from both sides

        decisive = len(wins) + len(losses)
        win_rate = len(wins) / decisive if decisive > 0 else 0.0

        gross_profit = wins.sum()  if len(wins)   > 0 else 0.0
        gross_loss   = abs(losses.sum()) if len(losses) > 0 else 0.0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")   # all-winners edge case
        else:
            profit_factor = 1.0

        avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0   # negative value

    # ── Time in Market ───────────────────────────────────────────────────────
    time_in_market = float(in_position.mean()) if len(in_position) > 0 else 0.0

    logger.info(
        "return=%.2f%% CAGR=%.2f%% maxDD=%.2f%% sharpe=%.2f sortino=%.2f "
        "calmar=%.2f winRate=%.1f%% PF=%.2f trades=%d timeInMkt=%.1f%%",
        total_return*100, cagr*100, max_dd*100, sharpe, sortino,
        calmar, win_rate*100, profit_factor, total_trades, time_in_market*100,
    )

    def _pf_fmt(pf: float) -> str:
        return "∞" if math.isinf(pf) else f"{pf:.2f}"

    def _calmar_fmt(c: float) -> str:
        return "∞" if math.isinf(c) else f"{c:.2f}"

    return {
        "Total Return":   {"fmt": f"{total_return:+.1%}", "raw": total_return},
        "CAGR":           {"fmt": f"{cagr:+.1%}",         "raw": cagr},
        "Max Drawdown":   {"fmt": f"{max_dd:.1%}",         "raw": max_dd},
        "Sharpe Ratio":   {"fmt": f"{sharpe:.2f}",         "raw": sharpe},
        "Sortino Ratio":  {"fmt": f"{sortino:.2f}",        "raw": sortino},
        "Calmar Ratio":   {"fmt": _calmar_fmt(calmar),     "raw": calmar},
        "Win Rate":       {"fmt": f"{win_rate:.1%}",       "raw": win_rate},
        "Profit Factor":  {"fmt": _pf_fmt(profit_factor),  "raw": profit_factor},
        "Avg Win ($)":    {"fmt": f"${avg_win:,.0f}",      "raw": avg_win},
        "Avg Loss ($)":   {"fmt": f"${avg_loss:,.0f}",     "raw": avg_loss},
        "Time in Market": {"fmt": f"{time_in_market:.1%}", "raw": time_in_market},
        "Total Trades":   {"fmt": str(total_trades),       "raw": total_trades},
    }


def build_portfolio_equity(ticker_results: dict) -> pd.Series:
    """
    Combine per-ticker equity curves into a single portfolio equity curve.

    Extracts daily PnL from each ticker's MTM equity, fills calendar gaps
    with zero (flat day, no position), sums all streams, and cumulates.
    """
    logger.info("Building portfolio equity from %d tickers", len(ticker_results))

    if not ticker_results:
        logger.warning("Empty ticker_results — returning empty series")
        return pd.Series(dtype=float)

    pnl_frames = []
    for name, eq in ticker_results.items():
        daily_pnl = eq.diff().fillna(0)
        logger.debug("%-20s total PnL=$%.2f", name, daily_pnl.sum())
        pnl_frames.append(daily_pnl)

    # fillna(0): trading holidays / different calendars → treat as flat day
    combined_pnl = pd.concat(pnl_frames, axis=1).fillna(0).sum(axis=1)

    start     = list(ticker_results.values())[0].iloc[0]
    portfolio = start + combined_pnl.cumsum()
    portfolio.name = "Portfolio Equity"

    logger.info(
        "Portfolio: start=$%.0f end=$%.0f return=%.2f%%",
        portfolio.iloc[0], portfolio.iloc[-1],
        (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100,
    )
    return portfolio
