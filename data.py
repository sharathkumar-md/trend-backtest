import logging

import pandas as pd
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

# Minimum rows needed: 100 (DMA_100 warmup) + 50 (breakout lookback) + buffer
_MIN_ROWS = 160


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Returns an empty DataFrame on any failure or if data is too thin
    (< 160 rows after cleaning), so callers can safely skip the ticker.

    Notes
    -----
    - auto_adjust=True: prices are split/dividend-adjusted as of download date.
    - MultiIndex columns (yfinance ≥ 0.2) are flattened defensively.
    - Rows with NaN in OHLC are dropped before the length check.
    """
    logger.info("Fetching %s (period=%s)", symbol, period)
    try:
        raw = yf.download(symbol, period=period, auto_adjust=True, progress=False)

        if raw.empty:
            logger.warning("%s — yfinance returned empty DataFrame", symbol)
            return pd.DataFrame()

        df = raw.copy()

        # ── Flatten MultiIndex columns (yfinance >= 0.2 behaviour) ────────────
        if isinstance(df.columns, pd.MultiIndex):
            # Level 0 is the field name ('Open', 'High', …), level 1 is ticker.
            # If level 0 does NOT look like OHLCV fields, try level 1.
            level0 = df.columns.get_level_values(0).tolist()
            if any(f in level0 for f in ("Open", "High", "Low", "Close")):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)
            logger.debug("%s — flattened MultiIndex columns", symbol)

        # ── Select OHLCV; synthesise Volume=0 if missing ─────────────────────
        ohlc = ["Open", "High", "Low", "Close"]
        missing = [c for c in ohlc if c not in df.columns]
        if missing:
            logger.warning("%s — missing columns %s, skipping", symbol, missing)
            return pd.DataFrame()

        df = df[ohlc].copy()
        df["Volume"] = raw.get("Volume", pd.Series(0, index=df.index))

        # ── Drop NaN OHLC rows (gaps, bad data) ──────────────────────────────
        before = len(df)
        df.dropna(subset=ohlc, inplace=True)
        if len(df) < before:
            logger.debug("%s — dropped %d NaN rows", symbol, before - len(df))

        if len(df) < _MIN_ROWS:
            logger.warning(
                "%s — only %d clean rows, need %d. Skipping.",
                symbol, len(df), _MIN_ROWS,
            )
            return pd.DataFrame()

        logger.info("%s — %d rows fetched OK", symbol, len(df))
        return df

    except Exception as exc:
        logger.error("Error fetching %s: %s", symbol, exc, exc_info=True)
        return pd.DataFrame()
