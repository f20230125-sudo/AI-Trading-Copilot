import pytest

from ai_trading_copilot.data import get_price_history
from ai_trading_copilot.signal_model import score_latest, train_signal_model


@pytest.fixture
def close_series():
    return get_price_history("DEMO", source="synthetic").frame["close"]


def test_train_signal_model_runs_and_returns_accuracy(close_series):
    model, val_accuracy = train_signal_model(close_series)
    assert model is not None
    assert val_accuracy is None or 0.0 <= val_accuracy <= 1.0


def test_score_latest_returns_valid_probability(close_series):
    result = score_latest(close_series)
    assert 0.0 <= result.probability_up <= 1.0
    assert result.predicted_direction in {"up", "down"}
    assert result.confidence in {"low", "medium", "high"}
    assert set(result.features_used.keys()) == {
        "rsi_14",
        "macd_hist",
        "bb_percent_b",
        "volatility_20d",
        "price_vs_sma20",
    }


def test_train_signal_model_raises_on_too_little_history():
    short_series = get_price_history("DEMO", source="synthetic").frame["close"].iloc[:20]
    with pytest.raises(ValueError):
        train_signal_model(short_series)


def test_same_ticker_is_deterministic_across_runs():
    a = get_price_history("AAPL", source="synthetic").frame["close"]
    b = get_price_history("AAPL", source="synthetic").frame["close"]
    assert a.equals(b)
