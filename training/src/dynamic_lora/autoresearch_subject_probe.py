"""Measure base-model layer gradients for tagged Autoresearch Manim subjects.
The probe makes no optimizer updates and keeps subject selection explicit."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.lora_parameters import lora_layer_key
from dynamic_lora.validation_gradient_heatmap import summarize_subject_layer_norms

DATASET_ID = "sebastianboehler/autoresearch-manim"
DATASET_REVISION = "31231ff601a0352872a1c6519faba7b97034f2b3"
SUBJECT_TAGS = {
    "math": "math",
    "mathematics": "math",
    "physics": "physics",
    "chemistry": "chemistry",
    "biology": "biology",
    "neuroscience": "neuroscience",
    "ml": "machine learning",
    "computer-science": "computer science",
    "cs": "computer science",
    "economics": "economics",
    "finance": "finance",
    "statistics": "statistics",
    "stats": "statistics",
}


def subject_for_tags(tags: list[str]) -> str | None:
    """Return one unambiguous subject, or None for untagged/mixed cases.

    Args:
        tags: Publisher-supplied free-form case tags.

    Returns:
        Canonical subject when exactly one is present.
    """
    subjects = {SUBJECT_TAGS[tag] for tag in tags if tag in SUBJECT_TAGS}
    return next(iter(subjects)) if len(subjects) == 1 else None


def capture_autoresearch_subject_probe(output_path: Path) -> dict[str, Any]:
    """Save per-subject mean base-weight gradient norms for all eligible cases.

    Args:
        output_path: JSON summary destination on the training artifact volume.

    Returns:
        Layer-by-subject norms and coverage/provenance details.

    Raises:
        ValueError: If the pinned dataset or model has unexpected structure.
    """
    datasets = import_module("datasets")
    torch = import_module("torch")
    transformers = import_module("transformers")
    cases = datasets.load_dataset(DATASET_ID, "cases", split="train", revision=DATASET_REVISION)
    if len(cases) != 150:
        raise ValueError("pinned Autoresearch release must have 150 distinct cases")
    selected = [(case, subject_for_tags(case["tags"])) for case in cases]
    eligible = [(case, subject) for case, subject in selected if subject is not None]
    if not eligible:
        raise ValueError("no cases have unambiguous subject tags")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID, revision=FROZEN_MODEL_REVISION, trust_remote_code=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(True)
    named = [
        (lora_layer_key(name).split(".")[0], parameter)
        for name, parameter in model.named_parameters()
        if lora_layer_key(name).split(".")[0].startswith("layer_")
    ]
    layers = {layer for layer, _ in named}
    if len(layers) != 36:
        raise ValueError("base model must have 36 transformer layers")

    rows: list[dict[str, Any]] = []
    lengths: dict[str, int] = {}
    for case, subject in eligible:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": case["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": case["prompt"]},
                {"role": "assistant", "content": case["completion"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise ValueError(f"chat template mismatch for {case['case_id']}")
        lengths[case["case_id"]] = len(full_ids)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        batch = {
            "input_ids": torch.tensor([full_ids], device=model.device),
            "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=model.device),
            "labels": torch.tensor([labels], device=model.device),
        }
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        squared: dict[str, float] = defaultdict(float)
        for layer, parameter in named:
            if parameter.grad is not None:
                squared[layer] += float(parameter.grad.detach().float().square().sum().item())
        if set(squared) != layers:
            raise ValueError(f"missing layer gradients for {case['case_id']}")
        rows.append(
            {
                "record_id": case["case_id"],
                "subject": subject,
                "layer_norms": {layer: math.sqrt(squared[layer]) for layer in layers},
            }
        )
        if len(rows) % 10 == 0:
            print(f"Measured {len(rows)}/{len(eligible)} eligible cases", flush=True)
    model.zero_grad(set_to_none=True)
    summary = summarize_subject_layer_norms(
        rows, expected_ids={case["case_id"] for case, _ in eligible}
    )
    summary.update(
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "gradient_source": "unadapted_base_transformer_layer_weights",
            "loss_target": "assistant_code_only",
            "optimizer_updates": 0,
            "source_count": len(cases),
            "excluded_untagged_or_ambiguous": len(cases) - len(eligible),
            "token_lengths": {"min": min(lengths.values()), "max": max(lengths.values())},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
