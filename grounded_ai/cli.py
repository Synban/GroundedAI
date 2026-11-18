"""Command-line entry points for the bare-minimum pipeline."""

from __future__ import annotations

import argparse

from grounded_ai.answer import answer_question, format_result
from grounded_ai.evaluate import main as evaluate_main
from grounded_ai.ingest import main as ingest_main
from grounded_ai.retrieve import main as index_main


def cmd_ask(args: argparse.Namespace) -> None:
    result = answer_question(args.question, top_k=args.top_k, min_score=args.min_score)
    print(format_result(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-ai", description="Grounded PDF QA")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Extract and chunk PDFs")
    sub.add_parser("index", help="Build TF-IDF retrieval index")
    sub.add_parser("eval", help="Run offline evaluation")

    ask = sub.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question", type=str, help="Natural language question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--min-score", type=float, default=0.05)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest_main()
    elif args.command == "index":
        index_main()
    elif args.command == "eval":
        evaluate_main()
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
