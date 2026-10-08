"""Prepare internal and chat-message training records from Bespoke-Manim data.
The script assigns reproducible identifiers and code-length difficulty proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

SYSTEM_PROMPT = (
    "You generate concise, runnable Manim Community Edition Python scenes for math tutoring. "
    "Return only Python code."
)


def main() -> None:
    """Convert one Parquet source into deterministic JSON Lines outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--internal-output", required=True, type=Path)
    parser.add_argument("--messages-output", required=True, type=Path)
    args = parser.parse_args()

    source_revision = "4542ab8b32483c30d1772946dacae2ad1ae9274c"
    rows = parquet.read_table(
        args.source,
        columns=["subject", "topic", "question", "python_code", "error", "scene_class_name"],
    ).to_pylist()
    valid = [row for row in rows if row["python_code"] and not row["error"]]
    ordered_lengths = sorted(len(str(row["python_code"]).splitlines()) for row in valid)
    lower = ordered_lengths[len(ordered_lengths) // 3]
    upper = ordered_lengths[2 * len(ordered_lengths) // 3]

    internal: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for index, row in enumerate(valid):
        code = str(row["python_code"]).strip()
        line_count = len(code.splitlines())
        difficulty = (
            "foundational"
            if line_count <= lower
            else "advanced"
            if line_count > upper
            else "intermediate"
        )
        stable_hash = hashlib.sha256(
            f"{row['subject']}\0{row['topic']}\0{row['question']}".encode()
        ).hexdigest()[:12]
        record_id = f"bespoke-manim-{index:04d}-{stable_hash}"
        user_prompt = f"Topic: {row['topic']}\nTask: {row['question']}"
        internal.append(
            {
                "id": record_id,
                "difficulty": difficulty,
                "topic": str(row["topic"]),
                "prompt": str(row["question"]),
                "manim_code": code,
                "split": "train",
                "source": {
                    "name": "bespokelabs/bespoke-manim",
                    "reference": f"{source_revision}:{index}",
                },
                "subject": str(row["subject"]),
                "scene_class_name": str(row["scene_class_name"]),
                "difficulty_proxy": "target-code-line-count-tertiles",
            }
        )
        messages.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": code},
                ]
            }
        )

    write_jsonl(args.internal_output, internal)
    write_jsonl(args.messages_output, messages)
    counts = {
        band: sum(row["difficulty"] == band for row in internal)
        for band in ("foundational", "intermediate", "advanced")
    }
    print(
        json.dumps(
            {
                "rows": len(internal),
                "skipped": len(rows) - len(internal),
                "difficulty_proxy_counts": counts,
            },
            sort_keys=True,
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to a UTF-8 JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
