# AI Trading Copilot

An LLM agent that answers natural-language questions about a ticker by
grounding its answer in real technical indicators, a small trained ML signal
model, and a retrieval layer over trading-concept documentation — rather than
answering from the model's own (unreliable, un-updatable) knowledge of
markets.

> **Portfolio / educational project.** Nothing here is financial advice, and
> the ML signal model makes no profitability claim — see [Risk Management](data/knowledge_base/risk_management.md).

## Why this exists

Built as a demonstration of an end-to-end AI engineering pipeline for
finance-adjacent products: tool-calling agent design, retrieval-augmented
generation, a properly time-split ML model (no lookahead leakage), and an
API/CLI surface — with every layer designed to degrade gracefully rather than
fail outright.

## Architecture

```
                    ┌───────────────────────┐
   User question ──▶│   Agent (Claude +      │
   + ticker         │   tool-calling loop)   │
                    └──────────┬─────────────┘
                               │ calls tools
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌────────────────┐ ┌─────────────┐ ┌──────────────────┐
     │ Price/indicator │ │ ML signal   │ │ RAG knowledge     │
     │ summary         │ │ model       │ │ base (TF-IDF)     │
     │ (yfinance or    │ │ (time-split │ │ over markdown     │
     │ synthetic)      │ │ GBM)        │ │ trading concepts  │
     └────────────────┘ └─────────────┘ └──────────────────┘
```

If `ANTHROPIC_API_KEY` is not set, the same tool pipeline still runs and
produces a deterministic, template-composed answer instead of an
LLM-composed one — the whole system works out of the box for anyone cloning
the repo, with no key required. If the LLM call fails at runtime for any
reason, the agent falls back to the same template path rather than erroring.

Market data itself has the same fallback shape: it tries `yfinance` (no API
key needed) and falls back to a deterministic, seeded synthetic price series
on any network failure or bad ticker, so tests, CI, and offline demos never
depend on Yahoo Finance being reachable.

## What's actually implemented

- **Technical indicators** (`src/ai_trading_copilot/indicators.py`): SMA,
  EMA, RSI, MACD, Bollinger %B, and annualized realized volatility, written
  directly on pandas and unit-tested against known values — not a black-box
  TA library.
- **ML signal model** (`signal_model.py`): a `GradientBoostingClassifier`
  predicting next-day direction from indicator features, trained with a
  **time-ordered** train/validation split so reported accuracy never peeks
  at the future.
- **RAG knowledge base** (`knowledge_base.py`): TF-IDF + cosine similarity
  retrieval over `data/knowledge_base/*.md` (indicator definitions, risk
  management). No embedding model download required; swapping in a
  sentence-transformer or hosted embedding API is a drop-in change behind
  `retrieve()`.
- **Agent** (`agent.py`): a real Claude tool-calling loop (`tools.py` defines
  the schemas) with a bounded number of turns and an offline template
  fallback.
- **API** (`api.py`): FastAPI app — `/health`, `/summary/{ticker}`,
  `/signal/{ticker}`, `/kb/search`, `/ask`.
- **CLI** (`cli.py`): `summary`, `signal`, and `ask` subcommands for a quick
  local demo without running the server.
- **Tests** (`tests/`): 20+ pytest tests covering indicator correctness
  against hand-computed values, model training/validation, RAG retrieval
  relevance, the agent's offline fallback, and the API endpoints — all
  forced onto the synthetic data path via `conftest.py` so the suite never
  depends on network access.
- **CI** (`.github/workflows/ci.yml`): lint (`ruff`) + test on Python 3.11
  and 3.12.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for LLM-composed answers
```

## Usage

CLI:

```bash
python -m ai_trading_copilot.cli summary AAPL
python -m ai_trading_copilot.cli signal AAPL
python -m ai_trading_copilot.cli ask AAPL "What does momentum look like right now?"
```

API server:

```bash
uvicorn ai_trading_copilot.api:app --reload
# then: curl "http://127.0.0.1:8000/signal/AAPL"
# curl -X POST http://127.0.0.1:8000/ask -H "content-type: application/json" \
#      -d '{"ticker": "AAPL", "question": "How is momentum looking?"}'
```

Tests:

```bash
pytest -v
ruff check src tests
```

## Design decisions worth calling out

- **Retraining per request** keeps the demo stateless and dependency-free
  (no model registry needed for a portfolio project); a production version
  would persist and periodically retrain the model instead.
- **TF-IDF over embeddings** for RAG: the knowledge base is small and
  domain-specific enough that lexical retrieval works well, and it avoids
  pulling in a multi-hundred-MB embedding model for five documents.
- **Every external dependency has a fallback path** (data source, LLM call):
  reliability under partial failure was treated as a first-class design
  constraint, not an afterthought.

## License

MIT — see [LICENSE](LICENSE).
