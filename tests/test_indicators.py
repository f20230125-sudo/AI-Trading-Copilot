import numpy as np
import pandas as pd
import pytest

from ai_trading_copilot.indicators import bollinger_bands, build_indicator_frame, ema, macd, rsi, sma


@pytest.fixture
def price_series() -> pd.Series:
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=100)))
    return pd.Series(prices, index=dates, name="close")


def test_sma_matches_manual_mean():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = sma(series, window=3)
    assert result.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    assert result.iloc[4] == pytest.approx((3 + 4 + 5) / 3)
    assert pd.isna(result.iloc[1])


def test_ema_reacts_faster_than_sma_of_same_span(price_series):
    ema_values = ema(price_series, span=10)
    sma_values = sma(price_series, window=10)
    # After a sustained move, EMA should sit closer to the latest price than SMA does.
    latest_price = price_series.iloc[-1]
    assert abs(ema_values.iloc[-1] - latest_price) <= abs(sma_values.iloc[-1] - latest_price) + 1e-9


def test_rsi_bounds(price_series):
    values = rsi(price_series).dropna()
    assert (values >= 0).all()
    assert (values <= 100).all()


def test_rsi_is_100_for_strictly_increasing_series():
    series = pd.Series(np.arange(1, 30, dtype=float))
    values = rsi(series, window=14).dropna()
    assert (values == 100).all()


def test_macd_histogram_equals_macd_minus_signal(price_series):
    result = macd(price_series)
    diff = (result["macd"] - result["signal"] - result["histogram"]).dropna()
    assert (diff.abs() < 1e-9).all()


def test_bollinger_percent_b_matches_definition(price_series):
    bb = bollinger_bands(price_series).dropna()
    expected = (price_series.loc[bb.index] - bb["lower"]) / (bb["upper"] - bb["lower"])
    assert np.allclose(bb["percent_b"], expected)


def test_build_indicator_frame_has_expected_columns(price_series):
    frame = build_indicator_frame(price_series)
    expected_columns = {
        "close",
        "sma_20",
        "sma_50",
        "ema_12",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_percent_b",
        "volatility_20d",
        "price_vs_sma20",
    }
    assert expected_columns.issubset(frame.columns)
    assert len(frame.dropna()) > 0
