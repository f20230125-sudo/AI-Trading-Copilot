"""Technical indicators computed from OHLCV data, implemented directly on pandas.

No third-party TA library — these are the standard textbook formulas, kept
small and unit-tested so the numbers can be trusted by the model and by
whoever reviews this repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100).astype(float)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    percent_b = (series - lower) / (upper - lower)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "percent_b": percent_b})


def realized_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Annualized volatility of daily log returns over a trailing window."""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std() * (252 ** 0.5)


def build_indicator_frame(close: pd.Series) -> pd.DataFrame:
    """All indicators aligned on one DataFrame, indexed like `close`."""
    frame = pd.DataFrame(index=close.index)
    frame["close"] = close
    frame["sma_20"] = sma(close, 20)
    frame["sma_50"] = sma(close, 50)
    frame["ema_12"] = ema(close, 12)
    frame["rsi_14"] = rsi(close, 14)

    macd_frame = macd(close)
    frame["macd"] = macd_frame["macd"]
    frame["macd_signal"] = macd_frame["signal"]
    frame["macd_hist"] = macd_frame["histogram"]

    bb = bollinger_bands(close)
    frame["bb_percent_b"] = bb["percent_b"]

    frame["volatility_20d"] = realized_volatility(close, 20)
    frame["price_vs_sma20"] = (close - frame["sma_20"]) / frame["sma_20"]

    return frame


def summarize_latest(frame: pd.DataFrame) -> dict:
    """Human-readable snapshot of the most recent row, for the agent/API to consume."""
    latest = frame.dropna().iloc[-1]
    return {
        "close": round(float(latest["close"]), 4),
        "sma_20": round(float(latest["sma_20"]), 4),
        "sma_50": round(float(latest["sma_50"]), 4),
        "rsi_14": round(float(latest["rsi_14"]), 2),
        "macd_hist": round(float(latest["macd_hist"]), 4),
        "bb_percent_b": round(float(latest["bb_percent_b"]), 4),
        "volatility_20d": round(float(latest["volatility_20d"]), 4),
        "trend": "above 50-day average" if latest["close"] > latest["sma_50"] else "below 50-day average",
        "rsi_state": (
            "overbought" if latest["rsi_14"] >= 70 else "oversold" if latest["rsi_14"] <= 30 else "neutral"
        ),
    }
