"""Command-line interface for the prover system."""

from __future__ import annotations

import argparse
from pathlib import Path

from prover.config import get_config, list_configs
from prover.runtime import run_problem


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prover", description="Multi-agent math prover."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the prover on an input file.")
    run_parser.add_argument("--config", required=True, help="Config name.")
    run_parser.add_argument(
        "--input", required=True, type=Path, help="Path to input Markdown file."
    )
    run_parser.add_argument(
        "--output", required=True, type=Path, help="Path to output Markdown file."
    )
    run_parser.add_argument(
        "--trace", type=Path, default=None, help="Optional JSONL trace path."
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the runtime max step cap for this run.",
    )

    subparsers.add_parser("list-configs", help="List available configurations.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-configs":
        for name in list_configs():
            print(name)
        return

    if args.command == "run":
        # Validate config
        get_config(args.config)
        run_problem(
            args.input,
            args.output,
            args.config,
            args.trace,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
