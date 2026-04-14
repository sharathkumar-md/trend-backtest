import logging

import numpy as np
import pandas as pd

from config import (
    ATR_PERIOD, DMA_FAST, DMA_SLOW, BREAKOUT_PERIOD, ATR_STOP_MULT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    True Wilder's ATR.

    Initialization: SMA of first `period` True-Range values.
    Subsequent bars: ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period

    This is Wilder's original formulation, NOT a simple EWM from bar 0,
    which would underweight early bars during the warmup phase.
    """
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    tr_arr = tr.values
    atr_arr = np.full(len(tr_arr), np.nan)

    if len(tr_arr) < period:
        logger.warning("Not enough rows (%d) to compute ATR(%d)", len(tr_arr), period)
        return pd.Series(atr_arr, index=df.index)

    # Seed: SMA of first `period` TR values (skip NaN from shift)
    seed_slice = tr_arr[:period]
    if np.any(np.isnan(seed_slice)):
        # First TR is NaN (no previous close), use bars 1..period+1
        seed_slice = tr_arr[1: period + 1]
        start_idx  = period + 1
    else:
        start_idx  = period

    atr_arr[start_idx - 1] = np.nanmean(seed_slice)

    # Wilder's smoothing
    for i in range(start_idx, len(tr_arr)):
        if not np.isnan(tr_arr[i]):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr_arr[i]) / period
        else:
            atr_arr[i] = atr_arr[i - 1]

    logger.debug("ATR(%d) computed. First valid index: %d", period, start_idx - 1)
    return pd.Series(atr_arr, index=df.index)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds strategy columns to df and drops leading NaN rows.

    Columns added
    -------------
    DMA_50, DMA_100  : simple moving averages of Close
    High_50, Low_50  : 50-day rolling high/low of *prior* day's High/Low
                       (shift(1) prevents lookahead — today's bar excluded)
    ATR              : Wilder's 20-day ATR
    Long_Filter      : DMA_50 > DMA_100
    Short_Filter     : DMA_50 < DMA_100
    Long_Signal      : Long_Filter AND today's Close > High_50 (breakout)
    Short_Signal     : Short_Filter AND today's Close < Low_50 (breakdown)
    """
    logger.debug("Calculating indicators on %d rows", len(df))
    df = df.copy()

    df["DMA_50"]  = df["Close"].rolling(DMA_FAST).mean()
    df["DMA_100"] = df["Close"].rolling(DMA_SLOW).mean()

    # shift(1) → only prior 50 bars count; today's bar is NOT included
    df["High_50"] = df["High"].shift(1).rolling(BREAKOUT_PERIOD).max()
    df["Low_50"]  = df["Low"].shift(1).rolling(BREAKOUT_PERIOD).min()

    df["ATR"] = calculate_atr(df, ATR_PERIOD)

    df["Long_Filter"]  = df["DMA_50"] > df["DMA_100"]
    df["Short_Filter"] = df["DMA_50"] < df["DMA_100"]

    df["Long_Signal"]  = df["Long_Filter"]  & (df["Close"] > df["High_50"])
    df["Short_Signal"] = df["Short_Filter"] & (df["Close"] < df["Low_50"])

    before = len(df)
    df.dropna(inplace=True)
    logger.debug(
        "Indicators done. Rows after dropna: %d (dropped %d warmup rows)",
        len(df), before - len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    portfolio_value: float = 1_000_000,
    risk_factor: float = 0.002,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Bar-by-bar backtest with mark-to-market equity and trailing stop.

    Execution model
    ---------------
    - Signals computed on day t's CLOSE data.
    - Trades executed at day t+1's OPEN (next-bar execution).
    - Each bar: exit check → stop update → entry check → MTM equity.
    - Equity is marked-to-market every bar (includes unrealised PnL).
    - Open position is force-closed at final bar's Close.

    Returns
    -------
    equity_series : pd.Series — mark-to-market portfolio value per day
    trades_df     : pd.DataFrame — one row per closed trade (incl. force-close)
    in_position   : pd.Series[bool] — True when a position is open
    """
    logger.info(
        "Backtest start: %d bars | portfolio=$%.0f | risk_factor=%.4f",
        len(df), portfolio_value, risk_factor,
    )

    opens         = df["Open"].values
    closes        = df["Close"].values
    atrs          = df["ATR"].values
    long_signals  = df["Long_Signal"].values
    short_signals = df["Short_Signal"].values
    n             = len(df)

    cash        = portfolio_value  # realised cash; grows only on trade close
    position    = 0               # 0=flat | 1=long | -1=short
    entry_price = 0.0
    units       = 0.0
    stop        = 0.0
    peak        = 0.0
    entry_date  = None

    equity_arr = np.empty(n, dtype=float)
    in_pos_arr = np.zeros(n, dtype=bool)
    trades     = []

    for i in range(n):
        open_price = opens[i]
        close      = closes[i]
        atr        = atrs[i]

        # ── 1. Exit check at today's open ────────────────────────────────────
        if position == 1 and open_price <= stop:
            pnl   = (open_price - entry_price) * units
            cash += pnl
            logger.debug(
                "LONG EXIT %s | open=%.4f stop=%.4f pnl=$%.2f",
                df.index[i].date(), open_price, stop, pnl,
            )
            trades.append(_trade_record(
                entry_date, df.index[i], "LONG",
                entry_price, open_price, units, pnl,
            ))
            position = 0

        elif position == -1 and open_price >= stop:
            pnl   = (entry_price - open_price) * units
            cash += pnl
            logger.debug(
                "SHORT EXIT %s | open=%.4f stop=%.4f pnl=$%.2f",
                df.index[i].date(), open_price, stop, pnl,
            )
            trades.append(_trade_record(
                entry_date, df.index[i], "SHORT",
                entry_price, open_price, units, pnl,
            ))
            position = 0

        # ── 2. Update trailing stop using today's CLOSE ──────────────────────
        #    (on entry bar this runs too — stop tracks from close of entry day)
        if position == 1:
            peak = max(peak, close)
            stop = peak - ATR_STOP_MULT * atr
        elif position == -1:
            peak = min(peak, close)
            stop = peak + ATR_STOP_MULT * atr

        # ── 3. Entry — flat only, signal from PRIOR day ──────────────────────
        if position == 0 and i > 0:
            prev_atr   = atrs[i - 1]
            prev_long  = long_signals[i - 1]
            prev_short = short_signals[i - 1]

            if prev_long and prev_atr > 0:
                # Size on MTM equity so compounding works correctly
                units       = (cash * risk_factor) / prev_atr
                entry_price = open_price
                position    = 1
                peak        = open_price
                stop        = open_price - ATR_STOP_MULT * prev_atr
                entry_date  = df.index[i]
                logger.debug(
                    "LONG ENTRY %s | open=%.4f units=%.4f stop=%.4f",
                    df.index[i].date(), open_price, units, stop,
                )

            elif prev_short and prev_atr > 0:
                units       = (cash * risk_factor) / prev_atr
                entry_price = open_price
                position    = -1
                peak        = open_price
                stop        = open_price + ATR_STOP_MULT * prev_atr
                entry_date  = df.index[i]
                logger.debug(
                    "SHORT ENTRY %s | open=%.4f units=%.4f stop=%.4f",
                    df.index[i].date(), open_price, units, stop,
                )

        # ── 4. Mark-to-market equity (unrealised PnL included) ───────────────
        if position == 1:
            unrealised = (close - entry_price) * units
        elif position == -1:
            unrealised = (entry_price - close) * units
        else:
            unrealised = 0.0

        equity_arr[i] = cash + unrealised
        in_pos_arr[i] = position != 0

    # ── 5. Force-close any open position at final bar's close ─────────────────
    if position != 0:
        final_close = closes[-1]
        if position == 1:
            pnl = (final_close - entry_price) * units
            direction = "LONG"
        else:
            pnl = (entry_price - final_close) * units
            direction = "SHORT"
        cash += pnl
        equity_arr[-1] = cash       # update final bar to realised value
        logger.info(
            "Force-closed %s position at end of data | pnl=$%.2f",
            direction, pnl,
        )
        rec = _trade_record(
            entry_date, df.index[-1], direction,
            entry_price, final_close, units, pnl,
        )
        rec["Note"] = "Force-closed (end of data)"
        trades.append(rec)

    equity_series = pd.Series(equity_arr, index=df.index, name="Equity")
    in_position   = pd.Series(in_pos_arr, index=df.index, name="InPosition")
    trades_df     = pd.DataFrame(trades)

    logger.info(
        "Backtest done: %d trades | final MTM=$%.2f | return=%.2f%%",
        len(trades_df), equity_arr[-1],
        (equity_arr[-1] / portfolio_value - 1) * 100,
    )
    return equity_series, trades_df, in_position


def _trade_record(
    entry_date, exit_date, direction,
    entry_price, exit_price, units, pnl,
) -> dict:
    return {
        "Entry Date":  entry_date,
        "Exit Date":   exit_date,
        "Direction":   direction,
        "Entry Price": round(entry_price, 4),
        "Exit Price":  round(exit_price, 4),
        "Units":       round(units, 4),
        "PnL ($)":     round(pnl, 2),
        "Note":        "",
    }


# ---------------------------------------------------------------------------
# Current signal helper
# ---------------------------------------------------------------------------

def get_current_signal(df: pd.DataFrame) -> dict:
    """
    Return today's actionable signal (based on yesterday's close data).
    Assumes df already has indicators calculated via calculate_indicators().
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    # Signal is from yesterday (prev), executed tomorrow at open
    if prev["Long_Signal"]:
        signal = "LONG"
    elif prev["Short_Signal"]:
        signal = "SHORT"
    else:
        signal = "FLAT"

    logger.debug(
        "Signal: %s | close=%.4f | DMA50=%.4f | DMA100=%.4f | ATR=%.4f",
        signal, last["Close"], last["DMA_50"], last["DMA_100"], last["ATR"],
    )

    return {
        "signal":       signal,
        "price":        round(float(last["Close"]), 4),
        "atr":          round(float(last["ATR"]), 4),
        "dma_50":       round(float(last["DMA_50"]), 4),
        "dma_100":      round(float(last["DMA_100"]), 4),
        "long_filter":  bool(last["Long_Filter"]),
        "short_filter": bool(last["Short_Filter"]),
        "high_50":      round(float(last["High_50"]), 4),
        "low_50":       round(float(last["Low_50"]), 4),
    }