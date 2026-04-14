import logging

import pandas as pd
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

# DMA_100 warmup (100) + breakout lookback (50) + small buffer = 120
# Kept deliberately low so cloud-hosted tickers with slightly fewer
# trading days (Indian / Asian markets) are not falsely rejected.
_MIN_ROWS = 120

# Fallback periods to try if the primary period returns too little data
_PERIODS = ["2y", "3y", "5y"]


def _clean(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten columns, select OHLCV, drop NaN rows."""
    df = raw.copy()

    # yfinance >= 0.2 may return MultiIndex(field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0).tolist()
        if any(f in level0 for f in ("Open", "High", "Low", "Close")):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
        logger.debug("%s — flattened MultiIndex columns", symbol)

    ohlc = ["Open", "High", "Low", "Close"]
    missing = [c for c in ohlc if c not in df.columns]
    if missing:
        logger.warning("%s — missing columns %s", symbol, missing)
        return pd.DataFrame()

    df = df[ohlc].copy()
    df["Volume"] = raw["Volume"] if "Volume" in raw.columns else 0

    before = len(df)
    df.dropna(subset=ohlc, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.debug("%s — dropped %d NaN rows", symbol, dropped)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Fetch 2 years of OHLCV data from yfinance.

    Retries with longer periods (3y, 5y) if 2y returns insufficient rows,
    then trims back to the last 2 years worth of trading days (~504 rows).
    This handles cloud environments where some exchanges (e.g. NSE) may
    return fewer rows for a given period string.

    Returns empty DataFrame on failure — callers should skip such tickers.
    """
    for period in _PERIODS:
        logger.info("Fetching %s (period=%s)", symbol, period)
        try:
            raw = yf.download(
                symbol, period=period,
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                logger.warning("%s — empty response for period=%s", symbol, period)
                continue

            df = _clean(raw, symbol)
            if df.empty:
                continue

            if len(df) < _MIN_ROWS:
                logger.warning(
                    "%s — only %d rows for period=%s (need %d), retrying …",
                    symbol, len(df), period, _MIN_ROWS,
                )
                continue

            # Trim to last ~504 trading days (≈ 2 years) after a wider fetch
            if len(df) > 504:
                df = df.iloc[-504:]

            logger.info("%s — %d rows OK (period=%s)", symbol, len(df), period)
            return df

        except Exception as exc:
            logger.error("Error fetching %s (period=%s): %s", symbol, period, exc)
            continue

    logger.error("%s — all periods failed, skipping ticker", symbol)
    return pd.DataFrame()