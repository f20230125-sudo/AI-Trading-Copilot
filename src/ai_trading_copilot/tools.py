"""Tool definitions the agent can call, in Anthropic tool-use schema.

Each schema is paired with a plain Python function so the same tools work
whether they're invoked by Claude's tool-calling loop or directly (the
fallback/offline agent path, or the REST API).
"""

from __future__ import annotations

import os

from .data import get_price_history
from .indicators import build_indicator_frame, summarize_latest
from .knowledge_base import KnowledgeBase
from .signal_model import score_latest

_kb = KnowledgeBase()

TOOL_SCHEMAS = [
    {
        "name": "get_price_summary",
        "description": "Get the latest close price and basic technical indicator snapshot for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL, BTC-USD."}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_signal_score",
        "description": (
            "Get the ML model's next-day direction probability for a ticker, derived from technical "
            "features. Not financial advice - a demonstration signal only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL, BTC-USD."}},
            "required": ["ticker"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the trading concepts knowledge base (indicator definitions, risk notes).",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Natural language search query."}},
            "required": ["query"],
        },
    },
]


def _data_source() -> str:
    """Override via AI_TRADING_COPILOT_DATA_SOURCE=synthetic to skip the network
    entirely (used in tests/CI so a Yahoo Finance outage never breaks a build)."""
    return os.environ.get("AI_TRADING_COPILOT_DATA_SOURCE", "auto")


def get_price_summary(ticker: str) -> dict:
    history = get_price_history(ticker, source=_data_source())
    frame = build_indicator_frame(history.frame["close"])
    summary = summarize_latest(frame)
    summary["ticker"] = history.ticker
    summary["data_source"] = history.source
    return summary


def get_signal_score(ticker: str) -> dict:
    history = get_price_history(ticker, source=_data_source())
    result = score_latest(history.frame["close"])
    return {
        "ticker": history.ticker,
        "data_source": history.source,
        "probability_up": result.probability_up,
        "predicted_direction": result.predicted_direction,
        "confidence": result.confidence,
        "validation_accuracy": result.validation_accuracy,
        "features_used": result.features_used,
    }


def search_knowledge_base(query: str) -> dict:
    docs = _kb.retrieve(query, k=2)
    return {"results": [doc.__dict__ for doc in docs]}


TOOL_IMPLEMENTATIONS = {
    "get_price_summary": get_price_summary,
    "get_signal_score": get_signal_score,
    "search_knowledge_base": search_knowledge_base,
}
