"""Measure held-out LoRA gradient norms by transformer layer and subject.
The module keeps dataset selection and aggregation separate from GPU execution."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import load_training_records, tokenize_completion_record
from dynamic_lora.full_corpus_plan import build_full_corpus_plan
from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key


def summarize_subject_layer_norms(
    rows: list[dict[str, Any]], *, expected_ids: set[str]
) -> dict[str, Any]:
    """Average per-example layer norms without weighting large subjects more.

    Args:
        rows: Gradient measurements with record ID, subject, and layer norms.
        expected_ids: Exact validation IDs that must be represented once.

    Returns:
        Subject counts and arithmetic mean norm for each observed layer.

    Raises:
        ValueError: If examples or layer measurements are incomplete.
    """
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("gradient rows must cover each validation ID exactly once")
    layer_keys = set(rows[0]["layer_norms"]) if rows else set()
    if not layer_keys:
        raise ValueError("gradient rows must contain layer norms")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        norms = row["layer_norms"]
        if set(norms) != layer_keys or any(
            not math.isfinite(value) or value < 0 for value in norms.values()
        ):
            raise ValueError("layer norms must be complete, finite, and nonnegative")
        grouped[str(row["subject"])].append(norms)
    return {
        "validation_count": len(rows),
        "subject_count": len(grouped),
        "layers": sorted(layer_keys, key=lambda value: int(value.removeprefix("layer_"))),
        "subjects": {
            subject: {
                "count": len(subject_rows),
                "mean_layer_norms": {
                    layer: sum(item[layer] for item in subject_rows) / len(subject_rows)
                    for layer in layer_keys
                },
            }
            for subject, subject_rows in sorted(grouped.items())
        },
    }


def capture_validation_gradient_heatmap(
    *, train_path: Path, checkpoint_path: Path, output_path: Path, seed: int = 42
) -> dict[str, Any]:
    """Probe the existing LoRA checkpoint on the disjoint validation split.

    Args:
        train_path: Original 995-row Manim JSONL source.
        checkpoint_path: Saved epoch-two PEFT checkpoint.
        output_path: Destination for aggregated JSON, distinct from training artifacts.
        seed: Seed used by the original validation split.

    Returns:
        Subject-by-layer mean norms with checkpoint provenance.

    Raises:
        ValueError: If the source split or checkpoint does not match expectations.
    """
    peft = import_module("peft")
    torch = import_module("torch")
    transformers = import_module("transformers")

    records = load_training_records(train_path)
    plan = build_full_corpus_plan(records, seed=seed, epochs=3)
    validation_ids = set(plan.validation_ids)
    if len(records) != 995 or len(validation_ids) != 100:
        raise ValueError("dataset no longer matches the 995/100 checkpoint split")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID, revision=FROZEN_MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    model.eval()
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if is_lora_parameter(name) and parameter.requires_grad
    ]
    keys = {lora_layer_key(name) for name, _ in named}
    layers = {key.split(".")[0] for key in keys}
    if len(layers) != 36 or len(keys) != 36 * 7:
        raise ValueError("checkpoint does not contain all 36 layers and seven LoRA projections")
    parameters = tuple(parameter for _, parameter in named)
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["id"] not in validation_ids:
            continue
        tokens = tokenize_completion_record(record, tokenizer, max_seq_length=3072)
        batch = {key: torch.tensor([value], device=model.device) for key, value in tokens.items()}
        with torch.enable_grad():
            loss = model(**batch).loss
            gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
        squared: dict[str, float] = defaultdict(float)
        for (name, _), gradient in zip(named, gradients, strict=True):
            layer = lora_layer_key(name).split(".")[0]
            squared[layer] += float(torch.sum(gradient.detach().float().square()).item())
        rows.append(
            {
                "record_id": record["id"],
                "subject": record["subject"],
                "layer_norms": {layer: math.sqrt(squared[layer]) for layer in layers},
            }
        )
        if len(rows) % 10 == 0:
            print(f"Measured {len(rows)}/{len(validation_ids)} validation examples", flush=True)
    summary = summarize_subject_layer_norms(rows, expected_ids=validation_ids)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "gradient_source": "exact_L2_of_LoRA_A_and_B_gradients_all_seven_projections",
            "loss_target": "assistant_code_only",
            "split_seed": seed,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def capture_base_validation_gradient_heatmap(
    *, train_path: Path, output_path: Path, seed: int = 42
) -> dict[str, Any]:
    """Measure unfine-tuned base-weight gradients on held-out examples.

    Args:
        train_path: Original 995-row Manim JSONL source.
        output_path: Destination for the separate base-gradient summary.
        seed: Seed used by the original validation split.

    Returns:
        Subject-by-layer mean base-weight gradient norms.

    Raises:
        ValueError: If the dataset or observed gradient layers are incomplete.
    """
    torch = import_module("torch")
    transformers = import_module("transformers")
    records = load_training_records(train_path)
    plan = build_full_corpus_plan(records, seed=seed, epochs=3)
    validation_ids = set(plan.validation_ids)
    if len(records) != 995 or len(validation_ids) != 100:
        raise ValueError("dataset no longer matches the 995/100 validation split")
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
    layer_parameters: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        if is_lora_parameter(name):
            continue
        layer_key = lora_layer_key(name).split(".")[0]
        if layer_key.startswith("layer_"):
            layer_parameters.append((layer_key, parameter))
    layers = {key for key, _ in layer_parameters}
    if len(layers) != 36:
        raise ValueError("base model does not contain all 36 transformer layers")
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["id"] not in validation_ids:
            continue
        tokens = tokenize_completion_record(record, tokenizer, max_seq_length=3072)
        batch = {key: torch.tensor([value], device=model.device) for key, value in tokens.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        squared: dict[str, float] = defaultdict(float)
        for layer, parameter in layer_parameters:
            gradient = parameter.grad
            if gradient is not None:
                squared[layer] += float(torch.sum(gradient.detach().float().square()).item())
        if set(squared) != layers:
            raise ValueError(f"base gradients missing layers for {record['id']}")
        rows.append(
            {
                "record_id": record["id"],
                "subject": record["subject"],
                "layer_norms": {layer: math.sqrt(squared[layer]) for layer in layers},
            }
        )
        if len(rows) % 10 == 0:
            print(f"Measured {len(rows)}/{len(validation_ids)} base examples", flush=True)
    model.zero_grad(set_to_none=True)
    summary = summarize_subject_layer_norms(rows, expected_ids=validation_ids)
    summary.update(
        {
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "gradient_source": "unadapted_base_transformer_layer_weights",
            "loss_target": "assistant_code_only",
            "optimizer_updates": 0,
            "split_seed": seed,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
