"""Command-line entrypoint for a quick local demo without running the API server.

Examples:
    python -m ai_trading_copilot.cli signal AAPL
    python -m ai_trading_copilot.cli ask AAPL "What does momentum look like right now?"
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import ask
from .tools import get_price_summary, get_signal_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-trading-copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Price + indicator snapshot")
    summary_parser.add_argument("ticker")

    signal_parser = subparsers.add_parser("signal", help="ML signal model score")
    signal_parser.add_argument("ticker")

    ask_parser = subparsers.add_parser("ask", help="Ask the agent a question about a ticker")
    ask_parser.add_argument("ticker")
    ask_parser.add_argument("question")

    args = parser.parse_args(argv)

    if args.command == "summary":
        print(json.dumps(get_price_summary(args.ticker), indent=2))
    elif args.command == "signal":
        print(json.dumps(get_signal_score(args.ticker), indent=2))
    elif args.command == "ask":
        response = ask(args.ticker, args.question)
        print(f"[mode: {response.mode}]\n")
        print(response.answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
