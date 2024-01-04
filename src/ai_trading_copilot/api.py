"""FastAPI service exposing the copilot over HTTP.

Run with: uvicorn ai_trading_copilot.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import ask
from .knowledge_base import KnowledgeBase
from .tools import get_price_summary, get_signal_score

app = FastAPI(
    title="AI Trading Copilot",
    description="LLM agent + ML signal model + RAG over trading concepts. Not financial advice.",
    version="0.1.0",
)

_kb = KnowledgeBase()


class AskRequest(BaseModel):
    ticker: str
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/signal/{ticker}")
def signal(ticker: str) -> dict:
    try:
        return get_signal_score(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/summary/{ticker}")
def summary(ticker: str) -> dict:
    return get_price_summary(ticker)


@app.get("/kb/search")
def kb_search(q: str, k: int = 3) -> dict:
    return {"results": [doc.__dict__ for doc in _kb.retrieve(q, k=k)]}


@app.post("/ask")
def ask_agent(request: AskRequest) -> dict:
    response = ask(request.ticker, request.question)
    return {"answer": response.answer, "mode": response.mode, "tool_calls": response.tool_calls}
