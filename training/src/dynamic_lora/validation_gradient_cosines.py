"""Compare held-out LoRA gradient directions at the saved model checkpoint.
The probe uses exact per-example gradients and applies subject labels only afterward."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import load_training_records, tokenize_completion_record
from dynamic_lora.full_corpus_plan import build_full_corpus_plan
from dynamic_lora.full_corpus_run import PROJECTION_CATEGORIES
from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key


def cosine_matrix(gram: np.ndarray) -> np.ndarray:
    """Normalize a Gram matrix into exact pairwise cosine similarities.

    Args:
        gram: Symmetric per-example gradient inner products.

    Returns:
        Cosines with NaN where either gradient has zero norm.

    Raises:
        ValueError: If the Gram matrix is malformed.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
        raise ValueError("gradient Gram matrix must be finite and square")
    lengths = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = lengths[:, None] * lengths[None, :]
    cosine = np.full_like(gram, np.nan, dtype=np.float64)
    np.divide(gram, denominator, out=cosine, where=denominator > 0)
    return np.clip(cosine, -1, 1)


def summarize_cosines(cosines: np.ndarray, subjects: list[str]) -> dict[str, Any]:
    """Summarize per-example cosines by observed subject without fitting groups.

    Args:
        cosines: Square, symmetric cosine matrix in example order.
        subjects: Metadata subject for each example in the same order.

    Returns:
        Cross-subject negative rates and subject-pair mean cosines.

    Raises:
        ValueError: If the matrix and subjects are inconsistent.
    """
    if cosines.shape != (len(subjects), len(subjects)) or not np.isfinite(cosines).all():
        raise ValueError("cosines must be complete and match example subjects")
    names = sorted(set(subjects))
    labels = np.asarray(subjects)
    mask = ~np.eye(len(subjects), dtype=bool)
    cross = mask & (labels[:, None] != labels[None, :])
    if not cross.any():
        raise ValueError("at least two subjects are needed")
    by_subject: dict[str, float] = {}
    pair_means: dict[str, dict[str, float | None]] = {}
    for subject in names:
        subject_rows = labels == subject
        involved = cross & subject_rows[:, None]
        by_subject[subject] = float(np.mean(cosines[involved] < 0))
        pair_means[subject] = {}
        for other in names:
            pair_mask = (subject_rows[:, None] & (labels == other)[None, :]) & mask
            pair_means[subject][other] = (
                float(np.mean(cosines[pair_mask])) if pair_mask.any() else None
            )
    return {
        "cross_subject_negative_rate": float(np.mean(cosines[cross] < 0)),
        "subject_negative_rate": by_subject,
        "subject_pair_mean_cosine": pair_means,
    }


def capture_validation_gradient_cosines(
    *, train_path: Path, checkpoint_path: Path, output_path: Path, seed: int = 42
) -> dict[str, Any]:
    """Capture exact LoRA gradient cosines for every held-out example and layer.

    Args:
        train_path: Original 995-row Manim source.
        checkpoint_path: Saved epoch-two PEFT adapter.
        output_path: Destination for aggregated cosine statistics.
        seed: Original validation-split seed.

    Returns:
        Layer and projection cosine summaries, plus subject sample counts.

    Raises:
        ValueError: If examples, adapter modules, or gradients are incomplete.
    """
    peft = import_module("peft")
    torch = import_module("torch")
    transformers = import_module("transformers")
    records = load_training_records(train_path)
    plan = build_full_corpus_plan(records, seed=seed, epochs=3)
    validation_ids = set(plan.validation_ids)
    examples = [record for record in records if record["id"] in validation_ids]
    if len(records) != 995 or len(examples) != 100 or {r["id"] for r in examples} != validation_ids:
        raise ValueError("dataset no longer matches the 995/100 validation split")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID, revision=FROZEN_MODEL_REVISION, trust_remote_code=True
    )
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
    if len(keys) != 36 * len(PROJECTION_CATEGORIES):
        raise ValueError("checkpoint lacks expected LoRA projection modules")
    parameters = tuple(parameter for _, parameter in named)
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, record in enumerate(examples, start=1):
        tokens = tokenize_completion_record(record, tokenizer, max_seq_length=3072)
        batch = {key: torch.tensor([value], device=model.device) for key, value in tokens.items()}
        with torch.enable_grad():
            gradients = torch.autograd.grad(model(**batch).loss, parameters, allow_unused=False)
        per_module: dict[str, list[np.ndarray]] = defaultdict(list)
        for (name, _), gradient in zip(named, gradients, strict=True):
            per_module[lora_layer_key(name)].append(
                gradient.detach().float().reshape(-1).cpu().numpy()
            )
        if set(per_module) != keys or any(len(parts) != 2 for parts in per_module.values()):
            raise ValueError(f"incomplete LoRA A/B gradients for {record['id']}")
        for key, parts in per_module.items():
            vectors[key].append(np.concatenate(parts))
        if index % 10 == 0:
            print(f"Captured {index}/{len(examples)} gradient vectors", flush=True)
    subjects = [str(record["subject"]) for record in examples]
    layer_grams = {
        f"layer_{index}": np.zeros((len(examples), len(examples))) for index in range(36)
    }
    modules: dict[str, Any] = {}
    for key in sorted(keys):
        matrix = np.stack(vectors[key]).astype(np.float64, copy=False)
        gram = matrix @ matrix.T
        layer = key.split(".")[0]
        layer_grams[layer] += gram
        modules[key] = summarize_cosines(cosine_matrix(gram), subjects)
        del vectors[key]
    layers = {
        layer: summarize_cosines(cosine_matrix(gram), subjects)
        for layer, gram in layer_grams.items()
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "gradient_source": "exact_LoRA_A_B_vectors",
        "loss_target": "assistant_code_only",
        "validation_count": len(examples),
        "subject_counts": dict(sorted(Counter(subjects).items())),
        "split_seed": seed,
        "layers": layers,
        "modules": modules,
        "aggregation": "mean_cross_example_cosine_by_subject_pair",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return summary
