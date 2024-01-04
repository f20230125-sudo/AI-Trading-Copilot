"""The agent: a Claude tool-calling loop over the market/RAG tools, with a
deterministic offline fallback so the whole pipeline runs without an API key.

Anyone cloning this repo without ANTHROPIC_API_KEY set still gets a real
answer built from the same tool calls (price data -> indicators -> ML signal
-> knowledge base retrieval); it's just template-assembled instead of
LLM-composed, and clearly labeled as such.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS, get_price_summary, get_signal_score, search_knowledge_base

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = (
    "You are a market analysis assistant for a trading education platform. "
    "You have tools to fetch price/indicator data, a trained signal model's probability, "
    "and a knowledge base of trading concept definitions. Always ground claims in tool "
    "output rather than prior knowledge about specific tickers. Always include a short "
    "disclaimer that this is not financial advice. Be concise."
)


@dataclass
class AgentResponse:
    answer: str
    mode: str  # "llm" or "offline-template"
    tool_calls: list[dict] = field(default_factory=list)


def _run_llm_agent(ticker: str, question: str) -> AgentResponse:
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Ticker: {ticker}\nQuestion: {question}"}]
    tool_calls: list[dict] = []

    for _ in range(4):  # hard cap so a misbehaving loop can't run away
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            return AgentResponse(answer=text.strip(), mode="llm", tool_calls=tool_calls)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOL_IMPLEMENTATIONS[block.name]
            result = tool_fn(**block.input)
            tool_calls.append({"name": block.name, "input": block.input, "output": result})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
            )
        messages.append({"role": "user", "content": tool_results})

    return AgentResponse(
        answer="The agent used too many tool calls without reaching a final answer.",
        mode="llm",
        tool_calls=tool_calls,
    )


def _run_offline_agent(ticker: str, question: str) -> AgentResponse:
    summary = get_price_summary(ticker)
    signal = get_signal_score(ticker)
    kb_hits = search_knowledge_base(question)["results"]

    tool_calls = [
        {"name": "get_price_summary", "input": {"ticker": ticker}, "output": summary},
        {"name": "get_signal_score", "input": {"ticker": ticker}, "output": signal},
        {"name": "search_knowledge_base", "input": {"query": question}, "output": kb_hits},
    ]

    kb_line = (
        f" Related concept: {kb_hits[0]['title']} ({kb_hits[0]['path']})." if kb_hits else ""
    )
    answer = (
        f"[offline template mode - set ANTHROPIC_API_KEY for LLM-composed answers]\n"
        f"{summary['ticker']} is trading at {summary['close']}, {summary['trend']}, "
        f"with RSI(14) at {summary['rsi_14']} ({summary['rsi_state']}) and "
        f"MACD histogram {summary['macd_hist']}. "
        f"The signal model puts the probability of an up move tomorrow at "
        f"{signal['probability_up']:.0%} ({signal['confidence']} confidence, "
        f"validated at {signal['validation_accuracy']:.0%} directional accuracy on held-out data)."
        f"{kb_line} "
        f"This is a portfolio project output, not financial advice."
    )
    return AgentResponse(answer=answer, mode="offline-template", tool_calls=tool_calls)


def ask(ticker: str, question: str) -> AgentResponse:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _run_llm_agent(ticker, question)
        except Exception as exc:  # noqa: BLE001 - degrade to offline mode on any API failure
            fallback = _run_offline_agent(ticker, question)
            fallback.answer = f"[LLM call failed ({exc}); falling back to offline mode]\n" + fallback.answer
            return fallback
    return _run_offline_agent(ticker, question)
