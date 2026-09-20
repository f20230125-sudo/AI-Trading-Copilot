from ai_trading_copilot.knowledge_base import KnowledgeBase


def test_retrieve_finds_rsi_doc_for_overbought_query():
    kb = KnowledgeBase()
    results = kb.retrieve("what does overbought RSI mean", k=1)
    assert len(results) == 1
    assert "RSI" in results[0].title


def test_retrieve_finds_risk_doc_for_position_sizing_query():
    kb = KnowledgeBase()
    results = kb.retrieve("how much of my capital should I risk per trade", k=1)
    assert len(results) == 1
    assert "Risk" in results[0].title


def test_retrieve_returns_nothing_for_unrelated_query():
    kb = KnowledgeBase()
    results = kb.retrieve("recipe for chocolate cake", k=2)
    assert results == []
