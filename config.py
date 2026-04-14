# Strategy constants
PORTFOLIO_SIZE  = 1_000_000   # $1M USD
RISK_FACTOR     = 0.002        # 20 bps per Clenow
ATR_PERIOD      = 20
DMA_FAST        = 50
DMA_SLOW        = 100
BREAKOUT_PERIOD = 50
ATR_STOP_MULT   = 3            # 3-ATR trailing stop
RISK_FREE_RATE  = 0.05         # 5% annualised

# Tickers by asset class — yfinance symbols
# Spec says skip if not working, so extras are fine
TICKERS = {
    "Equity": {
        "Nifty 50":    "^NSEI",
        "S&P 500":     "^GSPC",
        "FTSE 100":    "^FTSE",
        "Nikkei 225":  "^N225",
        "Hang Seng":   "^HSI",
    },
    "Currency": {
        "EUR/USD":  "EURUSD=X",
        "GBP/USD":  "GBPUSD=X",
        "USD/JPY":  "USDJPY=X",
        "USD/INR":  "USDINR=X",
        "USD/CHF":  "USDCHF=X",
        "USD/CNH":  "USDCNH=X",
    },
    "Commodity": {
        "Crude Oil":   "CL=F",
        "Natural Gas": "NG=F",
        "Gold":        "GC=F",
        "Silver":      "SI=F",
        "Copper":      "HG=F",
        "Soybean":     "ZS=F",
        "Corn":        "ZC=F",
        "Sugar":       "SB=F",
        "Coffee":      "KC=F",
        "Lumber":      "LBS=F",
    },
    "Bonds": {
        "US 10Y": "^TNX",
    },
    "Crypto": {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
    },
}

# Flat lookup: symbol -> (name, asset_class)
SYMBOL_MAP = {
    sym: (name, cls)
    for cls, tickers in TICKERS.items()
    for name, sym in tickers.items()
}

# Flat lookup: name -> symbol
NAME_TO_SYMBOL = {
    name: sym
    for tickers in TICKERS.values()
    for name, sym in tickers.items()
}

ALL_ASSET_CLASSES = ["All"] + list(TICKERS.keys())
