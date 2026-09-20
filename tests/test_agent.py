import pytest

from ai_trading_copilot.agent import ask


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    # Force the offline path so tests never depend on network/API credentials.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_ask_returns_offline_template_response_without_api_key():
    response = ask("DEMO", "What is the RSI telling us right now?")
    assert response.mode == "offline-template"
    assert "DEMO" in response.answer
    assert "not financial advice" in response.answer.lower()


def test_ask_records_tool_calls():
    response = ask("DEMO", "How is momentum looking?")
    tool_names = {call["name"] for call in response.tool_calls}
    assert tool_names == {"get_price_summary", "get_signal_score", "search_knowledge_base"}
