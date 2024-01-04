import pytest
from fastapi.testclient import TestClient

from ai_trading_copilot.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_endpoint():
    response = client.get("/summary/DEMO")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "DEMO"
    assert "rsi_14" in body


def test_signal_endpoint():
    response = client.get("/signal/DEMO")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_up"] <= 1.0


def test_kb_search_endpoint():
    response = client.get("/kb/search", params={"q": "overbought momentum"})
    assert response.status_code == 200
    assert len(response.json()["results"]) >= 1


def test_ask_endpoint_offline_mode():
    response = client.post("/ask", json={"ticker": "DEMO", "question": "How's the trend?"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "offline-template"
    assert "DEMO" in body["answer"]
