"""Provide command-line entrypoints for planning and running training experiments.
The CLI translates configuration paths into calls to the training modules."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_lora.config import load_config
from dynamic_lora.run_plan import build_run_plan, write_run_metadata
from dynamic_lora.training import train_shared_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Static shared-LoRA baseline for Manim SFT")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run", help="validate inputs and print a redacted run plan")
    dry_run.add_argument("--config", required=True, type=Path)

    train = subparsers.add_parser("train", help="run supervised fine-tuning for one shared LoRA")
    train.add_argument("--config", required=True, type=Path)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "dry-run":
        plan = build_run_plan(config)
        write_run_metadata(plan, config.metadata_path)
        print(plan.redacted_text)
        return

    if args.command == "train":
        plan = train_shared_lora(config)
        print(plan.redacted_text)
        return

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
