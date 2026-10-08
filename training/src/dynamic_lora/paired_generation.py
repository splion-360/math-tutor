"""Freeze a matched lesson evaluation and generate both model conditions.
CPU preflight records corpus overlap and token measurements before GPU execution."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from importlib import import_module, metadata
from pathlib import Path
from time import monotonic
from typing import Any, TypedDict, cast

from dynamic_lora.constants import ALLOWED_DIFFICULTIES, FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import (
    SYSTEM_PROMPT,
    format_training_prompt,
    load_training_records,
    tokenize_completion_record,
    validate_training_dataset,
)
from dynamic_lora.evidence import download_verified, file_sha256, verify_file

CONDITIONS = ("base", "shared_adapter")


class EvaluationPrompt(TypedDict):
    """Authored prompt and reference checks, excluding all generated outputs."""

    id: str
    difficulty: str
    topic: str
    prompt: str
    split: str
    source: dict[str, str]
    review_checks: list[str]


class FrozenPlan(TypedDict):
    """Immutable input contract shared by CPU preflight and GPU generation."""

    schema_version: int
    frozen_at_utc: str
    source_revision: str
    source_hashes: dict[str, str]
    model_id: str
    model_revision: str
    checkpoint: dict[str, Any]
    adapter_config: dict[str, Any]
    corpus_sha256: str
    corpus_count: int
    prompts_sha256: str
    examples: list[EvaluationPrompt]
    overlap_audit: list[dict[str, Any]]
    corpus_token_summary: dict[str, dict[str, int]]
    corpus_token_counts: list[dict[str, Any]]
    output_allowance_rule: str
    decoding: dict[str, Any]
    seed: int
    system_prompt: str
    user_template: str
    inference_precision: str
    condition_order: str
    scope: str
    grouping_note: str
    review_protocol: dict[str, str]
    renderer: dict[str, Any]
    limitations: list[str]


class GenerationRecord(TypedDict):
    """One raw first attempt with exact token counts and termination evidence."""

    id: str
    condition: str
    difficulty: str
    topic: str
    prompt: str
    response: str
    completion_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    generation_latency_seconds: float
    first_attempt: bool
    finish_reason: str
    output_capped: bool


def summarize_lengths(values: Sequence[int]) -> dict[str, int]:
    """Summarize nonempty token counts with nearest-rank percentiles.

    Args:
        values: Nonnegative integer token counts.

    Returns:
        Count, minimum, median, p95, p99, and maximum.

    Raises:
        ValueError: If counts are empty, negative, or nonintegers.
    """
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token counts must be nonempty nonnegative integers")
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": ordered[math.ceil(len(values) * 0.5) - 1],
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "p99": ordered[math.ceil(len(values) * 0.99) - 1],
        "max": ordered[-1],
    }


def completion_allowance(maximum: int) -> int:
    """Add 40 percent to the largest corpus target, rounded up to 256 tokens.

    Args:
        maximum: Largest measured assistant target, including its EOS token.

    Returns:
        Shared output allowance for both conditions.

    Raises:
        ValueError: If the measured maximum is not positive.
    """
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum completion length must be a positive integer")
    return math.ceil(maximum * 1.4 / 256) * 256


def audit_overlap(
    examples: Sequence[Mapping[str, Any]], corpus: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Reject ID/exact/near text overlap and record the nearest corpus prompt.

    Args:
        examples: Frozen prompt-only evaluation rows.
        corpus: Entire prepared corpus, including historical validation rows.

    Returns:
        Nearest word-trigram Jaccard matches for manual scrutiny.

    Raises:
        ValueError: If IDs or normalized prompts overlap, or trigram similarity is >=0.8.
    """

    def normalized(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).lower().split())

    def trigrams(text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"\w+", normalized(text))
        return {tuple(words[index : index + 3]) for index in range(len(words) - 2)}

    known_ids = {row["id"] for row in corpus}
    known_prompts = {normalized(row["prompt"]) for row in corpus}
    indexed = [(row, trigrams(row["prompt"])) for row in corpus]
    if not indexed:
        raise ValueError("overlap audit requires a nonempty corpus")
    audits = []
    seen: set[str] = set()
    seen_prompts: set[str] = set()
    for row in examples:
        prompt = normalized(row["prompt"])
        if row["id"] in known_ids or row["id"] in seen:
            raise ValueError("evaluation IDs overlap or repeat")
        if prompt in known_prompts or prompt in seen_prompts:
            raise ValueError("evaluation prompts overlap or repeat")
        seen.add(row["id"])
        seen_prompts.add(prompt)
        query = trigrams(prompt)
        best_score, nearest = max(
            (
                (len(query & grams) / len(query | grams) if query | grams else 1.0, item)
                for item, grams in indexed
            ),
            key=lambda pair: pair[0],
        )
        if best_score >= 0.8:
            raise ValueError(f"near-duplicate evaluation prompt: {row['id']}")
        audits.append(
            {
                "id": row["id"],
                "nearest_corpus_id": nearest["id"],
                "trigram_jaccard": best_score,
                "nearest_prompt": nearest["prompt"],
            }
        )
    return audits


def freeze_plan(
    *,
    corpus_path: Path,
    prompts_path: Path,
    evidence_path: Path,
    output: Path,
    tokenizer: Any,
    renderer_contract: dict[str, Any],
) -> FrozenPlan:
    """Freeze prompts, review criteria, decoding, and measured allowance before inference.

    Args:
        corpus_path: Hash-matching original prepared corpus.
        prompts_path: Fifteen project-authored prompts with reference review checks.
        evidence_path: Verified evidence release manifest.
        output: New JSON plan path; existing plans are never overwritten.
        tokenizer: Pinned research tokenizer.
        renderer_contract: Image, timeout, and evaluator/validator source hashes.

    Returns:
        Frozen plan written to output.

    Raises:
        ValueError: If source identities, prompts, or token measurements are invalid.
        OSError: If inputs cannot be read or output already exists.
    """
    evidence = json.loads(evidence_path.read_text())
    verify_file(corpus_path, evidence["dataset"]["prepared_sha256"])
    validate_training_dataset(corpus_path, prompts_path)
    corpus = load_training_records(corpus_path)
    examples = cast(list[EvaluationPrompt], load_training_records(prompts_path))
    counts = Counter(row.get("difficulty") for row in examples)
    if len(examples) != 15 or any(counts[key] != 5 for key in ALLOWED_DIFFICULTIES):
        raise ValueError("pilot requires five prompts in each author-assigned grouping")
    for row in examples:
        if (
            row.get("split") != "evaluation"
            or any(
                not isinstance(row.get(key), str) or not cast(str, row.get(key)).strip()
                for key in ("id", "topic", "prompt")
            )
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", row["id"])
        ):
            raise ValueError("pilot requires safe IDs and nonempty evaluation topics/prompts")
        checks = row.get("review_checks")
        if (
            not isinstance(checks, list)
            or not checks
            or any(not isinstance(check, str) or not check.strip() for check in checks)
        ):
            raise ValueError("every prompt requires frozen correctness checks")
    lengths: list[dict[str, Any]] = []
    for record in corpus:
        tokens = tokenize_completion_record(record, tokenizer, max_seq_length=3072)
        completion = sum(label != -100 for label in tokens["labels"])
        lengths.append(
            {
                "id": record["id"],
                "completion_tokens": completion,
                "prompt_tokens": len(tokens["input_ids"]) - completion,
                "total_tokens": len(tokens["input_ids"]),
            }
        )
    summary = {
        key: summarize_lengths([row[key] for row in lengths])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    source_dir = Path(__file__).parent
    source_hashes = {
        name: file_sha256(source_dir / name)
        for name in (
            "paired_generation.py",
            "data.py",
            "constants.py",
            "evidence.py",
        )
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    plan: FrozenPlan = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "source_hashes": source_hashes,
        "model_id": FROZEN_MODEL_ID,
        "model_revision": FROZEN_MODEL_REVISION,
        "checkpoint": evidence["checkpoint"],
        "adapter_config": next(
            a for a in evidence["artifacts"] if a["path"] == "adapter_config.json"
        ),
        "corpus_sha256": file_sha256(corpus_path),
        "corpus_count": len(corpus),
        "prompts_sha256": file_sha256(prompts_path),
        "examples": examples,
        "overlap_audit": audit_overlap(examples, corpus),
        "corpus_token_summary": summary,
        "corpus_token_counts": lengths,
        "output_allowance_rule": "ceil(maximum corpus completion * 1.4 / 256) * 256",
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "max_new_tokens": completion_allowance(summary["completion_tokens"]["max"]),
        },
        "seed": 42,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": "Topic: {topic}\nTask: {prompt}",
        "inference_precision": "BF16 base; original adapter with PEFT autocast_adapter_dtype=True",
        "condition_order": "alternate base-first and adapter-first by prompt index",
        "renderer": renderer_contract,
        "scope": "15 authored math prompts, one greedy first attempt per condition; silent videos",
        "grouping_note": "Author-assigned labels; not measured task difficulty or corpus tertiles.",
        "review_protocol": {
            "mathematical_correctness": "pass/fail/uncertain against frozen per-prompt checks",
            "prompt_adherence": "pass/fail/uncertain for requested content and visual operations",
            "duration_adherence": "video duration between 30 and 45 seconds, inclusive",
            "reviewer": "Human identity and notes required; agent assessments labeled separately",
            "missing_video": "unreviewable; do not treat code inspection as rendered-video review",
            "aggregation": (
                "Report counts over all 15 attempts; report reviewed-video counts separately"
            ),
        },
        "limitations": evidence["limitations"]
        + [
            "Lexical overlap checks do not establish semantic novelty or absence from pretraining.",
            "A small, authored math pilot does not estimate performance across the full dataset.",
            "The adapter was trained with 4-bit base weights; this comparison uses BF16 inference.",
            "One generation per prompt does not quantify sampling or training-seed variance.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def run_generations(
    *,
    plan_path: Path,
    output: Path,
    progress: Callable[[], None],
) -> None:
    """Generate paired raw completions on one GPU, saving each attempt immediately.

    Args:
        plan_path: Immutable CPU-preflight plan.
        output: New run directory on persistent artifact storage.
        progress: Flush persistent storage after each completed generation.

    Raises:
        ValueError: If source hashes differ from the frozen plan.
        OSError: If artifacts cannot be written or downloaded.
        RuntimeError: If model setup or GPU inference fails; completed outputs remain saved.
    """
    plan = cast(FrozenPlan, json.loads(plan_path.read_text()))
    for name, digest in plan["source_hashes"].items():
        verify_file(Path(__file__).parent / name, digest)
    torch, transformers, peft = (import_module(name) for name in ("torch", "transformers", "peft"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_plan.json").write_bytes(plan_path.read_bytes())
    progress()
    adapter = output / "adapter"
    download_verified(
        plan["checkpoint"]["url"],
        adapter / "adapter_model.safetensors",
        plan["checkpoint"]["sha256"],
    )
    download_verified(
        plan["adapter_config"]["url"],
        adapter / "adapter_config.json",
        plan["adapter_config"]["sha256"],
    )
    transformers.set_seed(plan["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        plan["model_id"],
        revision=plan["model_revision"],
        trust_remote_code=False,
    )
    started = monotonic()
    base = transformers.AutoModelForCausalLM.from_pretrained(
        plan["model_id"],
        revision=plan["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model = peft.PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    load_seconds = monotonic() - started
    warmup = tokenizer("Return Python code: print(1)", return_tensors="pt").to("cuda")
    started = monotonic()
    for condition in CONDITIONS:
        context = model.disable_adapter() if condition == "base" else nullcontext()
        with context, torch.inference_mode():
            model.generate(
                **warmup, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
    torch.cuda.synchronize()
    runtime = {
        "model_load_seconds": load_seconds,
        "warmup_seconds": monotonic() - started,
        "warmup": "One token in each condition on a non-evaluation prompt; excluded from attempts",
        "gpu": torch.cuda.get_device_name(),
        "plan_sha256": file_sha256(plan_path),
        "packages": {
            name: metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            )
        },
        "generation_settings": plan["decoding"],
        "status": "generating",
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    progress()
    for index, example in enumerate(plan["examples"]):
        prompt = format_training_prompt(cast(dict[str, Any], example), tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        conditions = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in conditions:
            context = model.disable_adapter() if condition == "base" else nullcontext()
            with context, torch.inference_mode():
                enabled = [layer.enabled for layer in model.get_layer_status()]
                if not enabled or any(
                    value != (condition == "shared_adapter") for value in enabled
                ):
                    raise RuntimeError("adapter state does not match the requested condition")
                transformers.set_seed(plan["seed"])
                torch.cuda.synchronize()
                started = monotonic()
                generation_config = transformers.GenerationConfig(
                    **plan["decoding"],
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated = model.generate(**inputs, generation_config=generation_config)
                torch.cuda.synchronize()
                elapsed = monotonic() - started
            completion = generated[0, inputs["input_ids"].shape[1] :].tolist()
            ended = bool(completion and completion[-1] == tokenizer.eos_token_id)
            row: GenerationRecord = {
                "id": example["id"],
                "condition": condition,
                "difficulty": example["difficulty"],
                "topic": example["topic"],
                "prompt": example["prompt"],
                "response": tokenizer.decode(completion, skip_special_tokens=True),
                "completion_token_ids": completion,
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(completion),
                "generation_latency_seconds": elapsed,
                "first_attempt": True,
                "finish_reason": "eos" if ended else "length",
                "output_capped": not ended
                and len(completion) >= plan["decoding"]["max_new_tokens"],
            }
            with (output / "generations.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            progress()
            print(
                f"{condition} {example['id']}: {len(completion)} tokens, {elapsed:.1f}s", flush=True
            )
    runtime["status"] = "completed"
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    progress()


def main() -> None:
    """Run the CPU-only freeze step with the pinned research tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("training/artifacts/evidence/bespoke_manim_train.jsonl")
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path("backend/data/evaluation/manim_paired_pilot_v2.jsonl")
    )
    parser.add_argument("--evidence", type=Path, default=Path("training/evidence/manifest.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("training/artifacts/paired-pilot-v2/frozen_plan.json")
    )
    args = parser.parse_args()
    transformers = import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        trust_remote_code=False,
        cache_dir="training/artifacts/tokenizer-audit",
    )
    plan = freeze_plan(
        corpus_path=args.corpus,
        prompts_path=args.prompts,
        evidence_path=args.evidence,
        output=args.output,
        tokenizer=tokenizer,
        renderer_contract=import_module("math_tutor.paired_evaluation").renderer_contract(),
    )
    print(
        json.dumps(
            {
                "summary": plan["corpus_token_summary"],
                "decoding": plan["decoding"],
                "plan_sha256": file_sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
