import pytest


@pytest.fixture(autouse=True)
def force_synthetic_data(monkeypatch):
    """Keep the whole test suite offline and deterministic: no test should
    depend on Yahoo Finance being reachable or on live market data changing."""
    monkeypatch.setenv("AI_TRADING_COPILOT_DATA_SOURCE", "synthetic")
