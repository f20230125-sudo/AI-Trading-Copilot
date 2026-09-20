"""Market data loading with an offline-safe synthetic fallback.

Real deployments should hit a live market data feed (yfinance here, since it
needs no API key). CI environments and offline demos often can't reach
Yahoo Finance, so every ticker also has a deterministic synthetic path that
never touches the network. The two paths return the same schema so the rest
of the pipeline doesn't care which one served the request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Source = Literal["auto", "yfinance", "synthetic"]


@dataclass
class PriceHistory:
    ticker: str
    source: str
    frame: pd.DataFrame  # columns: open, high, low, close, volume; DatetimeIndex


def _seed_for_ticker(ticker: str) -> int:
    digest = hashlib.sha256(ticker.upper().encode()).hexdigest()
    return int(digest[:8], 16)


def _synthetic_history(ticker: str, periods: int = 180, start_price: float = 100.0) -> pd.DataFrame:
    """Deterministic geometric-random-walk OHLCV series, seeded by ticker name.

    Same ticker always produces the same series, which keeps tests and demos
    reproducible without pinning a data file.
    """
    rng = np.random.default_rng(_seed_for_ticker(ticker))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    drift = 0.0002
    vol = 0.018
    shocks = rng.normal(loc=drift, scale=vol, size=periods)
    close = start_price * np.exp(np.cumsum(shocks))

    intraday_spread = np.abs(rng.normal(loc=0.006, scale=0.003, size=periods))
    high = close * (1 + intraday_spread)
    low = close * (1 - intraday_spread)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(low=1_000_000, high=8_000_000, size=periods)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _yfinance_history(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).history(period=period, interval="1d")
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        raw.index.name = "date"
        return raw
    except Exception:
        return None


def get_price_history(ticker: str, period: str = "6mo", source: Source = "auto") -> PriceHistory:
    """Fetch daily OHLCV data for `ticker`.

    source="auto" tries yfinance first and silently falls back to the
    synthetic generator on any failure (offline, rate-limited, bad ticker).
    source="synthetic" skips the network entirely, which is what tests and
    CI use so a Yahoo Finance outage never breaks the build.
    """
    if source in ("auto", "yfinance"):
        frame = _yfinance_history(ticker, period)
        if frame is not None and len(frame) >= 30:
            return PriceHistory(ticker=ticker.upper(), source="yfinance", frame=frame)
        if source == "yfinance":
            raise ValueError(f"No yfinance data available for {ticker!r}")

    return PriceHistory(ticker=ticker.upper(), source="synthetic", frame=_synthetic_history(ticker))
