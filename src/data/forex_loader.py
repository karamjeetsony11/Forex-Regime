"""Yahoo Finance hourly FX candle loader (legacy module path)."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Yahoo Finance's public chart endpoint provides hourly spot FX candles.
# Symbols accepted by the app are six-letter currency pairs, e.g. EURUSD.
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
DEFAULT_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
_INTERVAL_SECONDS = {"1h": 3600, "60m": 3600, "30m": 1800, "15m": 900, "5m": 300, "1d": 86400}


class ForexDataLoader:
    """Fetch OHLC candles for major FX pairs from Yahoo Finance."""

    def __init__(self, symbols: Optional[list[str]] = None, base_url: Optional[str] = None,
                 output_dir: Optional[str] = None) -> None:
        self.symbols = [self.normalize_symbol(s) for s in (symbols or DEFAULT_SYMBOLS)]
        self.output_dir = output_dir or DEFAULT_RAW_DIR
        self.base_url = (base_url or os.environ.get("YAHOO_FINANCE_API_URL", "https://query1.finance.yahoo.com")).rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 market-regime-detector/1.0"})
        logger.info("ForexDataLoader created — pairs=%s", self.symbols)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        pair = re.sub(r"[^A-Za-z]", "", symbol).upper()
        if len(pair) == 7 and pair.endswith("X"):
            pair = pair[:6]
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError("FX symbol must be a six-letter currency pair, e.g. EURUSD.")
        return pair

    def fetch_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
        pair = self.normalize_symbol(symbol)
        yahoo_interval = {"1h": "60m", "60m": "60m"}.get(interval, interval)
        if yahoo_interval not in _INTERVAL_SECONDS:
            raise ValueError(f"Unsupported FX interval {interval!r}; use 1h, 30m, 15m, 5m, or 1d.")
        if limit < 31:
            raise ValueError("At least 31 candles are required for the feature warm-up.")
        now = int(time.time())
        period = min(max(limit * _INTERVAL_SECONDS[yahoo_interval] * 2, 7 * 86400), 59 * 86400 if yahoo_interval != "1d" else 365 * 86400)
        url = f"{self.base_url}/v8/finance/chart/{pair}=X"
        response = self._session.get(url, params={"period1": now - period, "period2": now, "interval": yahoo_interval, "events": "history"}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        chart = payload.get("chart", {})
        if chart.get("error"):
            raise RuntimeError(f"Yahoo Finance rejected {pair}: {chart['error'].get('description', 'unknown error')}")
        result = (chart.get("result") or [None])[0]
        if not result:
            raise RuntimeError(f"No Yahoo Finance candles returned for {pair}.")
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        df = pd.DataFrame({
            "open_time": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open", []), "high": quote.get("high", []),
            "low": quote.get("low", []), "close": quote.get("close", []),
        })
        df = df.dropna(subset=["open", "high", "low", "close"]).tail(min(limit, 1000)).reset_index(drop=True)
        if len(df) < 31:
            raise RuntimeError(f"Only {len(df)} valid candles returned for {pair}; need at least 31.")
        # FX has no consolidated volume or exchange order book. Neutral values
        # preserve the legacy feature schema while candle range/body provide
        # the actual price-derived features used by the FX-trained model.
        df["volume"] = 1.0
        df["quote_volume"] = df["close"]
        df["trades"] = 0
        df["taker_buy_base_vol"] = 0.5
        df["taker_buy_quote_vol"] = df["close"] * 0.5
        logger.info("Fetched %d hourly FX candles for %s", len(df), pair)
        return df

    def fetch_trades(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_orderbook(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_all(self, interval: str = "1h", ohlcv_limit: int = 500,
                  trades_limit: int = 500, orderbook_limit: int = 100) -> dict:
        result = {}
        for symbol in self.symbols:
            candles = self.fetch_ohlcv(symbol, interval=interval, limit=ohlcv_limit)
            result[symbol] = {"ohlcv": candles}
            self.save_to_csv(candles, symbol, "ohlcv", extra_tag=interval)
        return result

    def save_to_csv(self, df: pd.DataFrame, symbol: str, data_type: str,
                    extra_tag: Optional[str] = None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        tag = f"_{extra_tag}" if extra_tag else ""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{self.normalize_symbol(symbol)}_{data_type}{tag}_{stamp}.csv")
        df.to_csv(path, index=False)
        return path
