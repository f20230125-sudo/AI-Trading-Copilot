"""A small, honest ML model: predicts next-day direction from technical features.

This is not a trading strategy and makes no profitability claim - it is here
to show a real supervised-learning pipeline (feature engineering, a
time-ordered train/validation split to avoid lookahead leakage, and a
calibrated probability output) feeding into the agent as one more signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from .indicators import build_indicator_frame

FEATURE_COLUMNS = [
    "rsi_14",
    "macd_hist",
    "bb_percent_b",
    "volatility_20d",
    "price_vs_sma20",
]


@dataclass
class SignalResult:
    probability_up: float
    predicted_direction: str
    confidence: str
    validation_accuracy: float | None
    features_used: dict


def _build_training_frame(close: pd.Series) -> pd.DataFrame:
    frame = build_indicator_frame(close)
    frame["next_return"] = close.shift(-1) / close - 1
    frame["label_up"] = (frame["next_return"] > 0).astype(int)
    return frame.dropna(subset=[*FEATURE_COLUMNS, "label_up"])


def train_signal_model(
    close: pd.Series, validation_fraction: float = 0.2
) -> tuple[GradientBoostingClassifier, float | None]:
    """Time-ordered split: train on the earlier segment, validate on the later
    one, so the reported accuracy never peeks at the future."""
    frame = _build_training_frame(close)
    if len(frame) < 40:
        raise ValueError("Not enough history to train a signal model (need >= 40 usable rows)")

    split_idx = int(len(frame) * (1 - validation_fraction))
    train_frame, val_frame = frame.iloc[:split_idx], frame.iloc[split_idx:]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["label_up"])

    val_accuracy = None
    if len(val_frame) > 0:
        preds = model.predict(val_frame[FEATURE_COLUMNS])
        val_accuracy = float(accuracy_score(val_frame["label_up"], preds))

    return model, val_accuracy


def score_latest(close: pd.Series) -> SignalResult:
    """Train on all available history and score the most recent bar.

    Retraining per request keeps the demo stateless and dependency-free; a
    production version would persist and periodically retrain the model
    instead of refitting on every call.
    """
    model, val_accuracy = train_signal_model(close)
    frame = build_indicator_frame(close).dropna(subset=FEATURE_COLUMNS)
    latest_features = frame.iloc[[-1]][FEATURE_COLUMNS]

    probability_up = float(model.predict_proba(latest_features)[0][1])
    direction = "up" if probability_up >= 0.5 else "down"
    distance_from_even = abs(probability_up - 0.5)
    confidence = "low" if distance_from_even < 0.08 else "medium" if distance_from_even < 0.18 else "high"

    return SignalResult(
        probability_up=round(probability_up, 4),
        predicted_direction=direction,
        confidence=confidence,
        validation_accuracy=round(val_accuracy, 4) if val_accuracy is not None else None,
        features_used=latest_features.iloc[0].round(4).to_dict(),
    )
