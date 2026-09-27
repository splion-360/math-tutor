"""Run a no-update Qwen base-weight probe with the shared Manim data contract.
This module owns probe configuration, provenance checks, and saved experiment results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from dynamic_lora.base_layer_probe import (
    compare_probe_rankings,
    measure_base_layer_gradient_energy,
    measure_lora_subset_gradient_energy,
    selection_frequency,
)
from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import (
    format_training_record,
    load_training_records,
    tokenize_training_batch,
    validate_training_dataset,
)


@dataclass(frozen=True)
class BaseProbeConfig:
    """Inputs, artifacts, and tracking identity for a BF16 base-gradient probe."""

    train_path: Path
    holdout_path: Path
    metadata_path: Path
    reference_metadata_path: Path
    sample_count: int
    top_k: int
    max_seq_length: int
    target_modules: tuple[str, ...]
    wandb_project: str
    wandb_run_name: str
    modal_artifact_path: str
    seed: int = 42


class BaseProbeConfigError(ValueError):
    """Report an invalid no-update base-probe configuration."""


def load_base_probe_config(path: Path) -> BaseProbeConfig:
    """Parse and validate the dedicated base-gradient probe configuration.

    Args:
        path: JSON config path; relative artifact paths resolve from its directory.

    Returns:
        Validated configuration for a no-update BF16 probe.

    Raises:
        BaseProbeConfigError: If any required value has an invalid type or range.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BaseProbeConfigError("config root must be an object")

    def string(name: str) -> str:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise BaseProbeConfigError(f"{name} must be a non-empty string")
        return value

    def location(name: str) -> Path:
        value = Path(string(name))
        return value if value.is_absolute() else path.parent / value

    def positive_integer(name: str, minimum: int = 1) -> int:
        value = raw.get(name)
        if type(value) is not int or value < minimum:
            raise BaseProbeConfigError(f"{name} must be an integer >= {minimum}")
        return value

    target_modules = raw.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(module, str) and module for module in target_modules)
        or len(set(target_modules)) != len(target_modules)
    ):
        raise BaseProbeConfigError("target_modules must be distinct non-empty strings")

    return BaseProbeConfig(
        train_path=location("train_path"),
        holdout_path=location("holdout_path"),
        metadata_path=location("metadata_path"),
        reference_metadata_path=location("reference_metadata_path"),
        sample_count=positive_integer("sample_count", 2),
        top_k=positive_integer("top_k"),
        max_seq_length=positive_integer("max_seq_length", 2),
        target_modules=tuple(target_modules),
        wandb_project=string("wandb_project"),
        wandb_run_name=string("wandb_run_name"),
        modal_artifact_path=string("modal_artifact_path"),
        seed=positive_integer("seed") if "seed" in raw else 42,
    )


def validate_reference_probe(
    reference: dict[str, Any],
    *,
    model_id: str,
    model_revision: str,
    training_content_sha256: str,
    holdout_content_sha256: str,
) -> list[str]:
    """Return old top modules only when their model and data match this probe.

    Args:
        reference: Previous shared-LoRA run metadata.
        model_id: Frozen base-model identifier.
        model_revision: Frozen base-model revision.
        training_content_sha256: Current training-data fingerprint.
        holdout_content_sha256: Current holdout-data fingerprint.

    Returns:
        Previously selected LoRA module keys in rank order.

    Raises:
        ValueError: If identity or data differ, or the reference lacks rankings.
    """
    if reference.get("model_id") != model_id or reference.get("model_revision") != model_revision:
        raise ValueError("reference model does not match base probe")
    dataset = reference.get("dataset", {})
    if dataset.get("training_content_sha256") != training_content_sha256:
        raise ValueError("reference training dataset does not match base probe")
    if dataset.get("holdout_content_sha256") != holdout_content_sha256:
        raise ValueError("reference holdout dataset does not match base probe")
    old_probe = reference.get("layer_energy_probe", {})
    selected = old_probe.get("selected_layers")
    if not isinstance(selected, list) or not selected or not all(
        isinstance(key, str) for key in selected
    ):
        raise ValueError("reference has no selected LoRA modules")
    return selected


def run_base_probe(config: BaseProbeConfig) -> dict[str, Any]:
    """Probe unfrozen BF16 base weights, then save a ranked comparison.

    Args:
        config: Validated probe inputs and output locations.

    Returns:
        Result metadata including rankings, stability, comparison, and W&B URL.

    Raises:
        RuntimeError: If W&B access is missing or the model cannot run the probe.
        ValueError: If the reference experiment is not comparable.
    """
    report = validate_training_dataset(config.train_path, config.holdout_path)
    reference = json.loads(config.reference_metadata_path.read_text(encoding="utf-8"))
    old_modules = validate_reference_probe(
        reference,
        model_id=FROZEN_MODEL_ID,
        model_revision=FROZEN_MODEL_REVISION,
        training_content_sha256=report.training_content_sha256,
        holdout_content_sha256=report.holdout_content_sha256,
    )
    if reference["layer_energy_probe"].get("sample_count") != config.sample_count:
        raise ValueError("reference sample count does not match base probe")
    reference_training = reference.get("training", {})
    if (
        reference_training.get("max_seq_length") != config.max_seq_length
        or reference_training.get("seed") != config.seed
    ):
        raise ValueError("reference token limit or seed does not match base probe")
    lora_settings = reference.get("lora", {})
    if tuple(lora_settings.get("target_modules", ())) != config.target_modules:
        raise ValueError("reference LoRA target modules do not match base probe")
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY is required for the base probe")

    datasets = import_module("datasets")
    peft = import_module("peft")
    torch = import_module("torch")
    transformers = import_module("transformers")
    wandb = import_module("wandb")
    transformers.set_seed(config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_training_records(config.train_path)
    dataset = datasets.Dataset.from_list(
        [format_training_record(record, tokenizer) for record in records]
    )
    tokenized = dataset.map(
        lambda batch: tokenize_training_batch(
            batch, tokenizer, max_seq_length=config.max_seq_length
        ),
        batched=True,
        remove_columns=["text"],
    )
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.requires_grad_(True)

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        job_type="base-gradient-probe",
        config={
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "gradient_source": "base_weights",
            "precision": "bf16",
            "optimizer_steps": 0,
            "sample_count": config.sample_count,
            "top_k": config.top_k,
            "max_seq_length": config.max_seq_length,
            "target_modules": list(config.target_modules),
            "training_content_sha256": report.training_content_sha256,
            "reference_run_url": reference.get("tracking", {}).get("run_url"),
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
        },
    )
    try:
        probe = measure_base_layer_gradient_energy(
            model=model,
            tokenized_dataset=tokenized,
            data_collator=collator,
            sample_count=config.sample_count,
            top_k=config.top_k,
            target_modules=config.target_modules,
        )
        group_count = min(4, len(probe["sample_indices"]))
        stability_groups: list[dict[str, Any]] = []
        for offset in range(group_count):
            source_indices = probe["sample_indices"][offset::group_count]
            group_probe = measure_base_layer_gradient_energy(
                model=model,
                tokenized_dataset=tokenized.select(source_indices),
                data_collator=collator,
                sample_count=len(source_indices),
                top_k=config.top_k,
                target_modules=config.target_modules,
            )
            stability_groups.append(
                {
                    "source_indices": source_indices,
                    "selected_modules": group_probe["selected_modules"],
                    "selected_layers": group_probe["selected_layers"],
                }
            )
        module_frequency = selection_frequency(
            [group["selected_modules"] for group in stability_groups]
        )
        layer_frequency = selection_frequency(
            [group["selected_layers"] for group in stability_groups]
        )
        historical_comparison = compare_probe_rankings(
            base_modules=probe["selected_modules"], lora_modules=old_modules
        )
        model.zero_grad(set_to_none=True)
        lora_config = peft.LoraConfig(
            r=lora_settings["r"],
            lora_alpha=lora_settings["alpha"],
            lora_dropout=lora_settings["dropout"],
            target_modules=list(config.target_modules),
            task_type=peft.TaskType.CAUSAL_LM,
        )
        adapted_model = peft.get_peft_model(model, lora_config)
        matched_lora_probe = measure_lora_subset_gradient_energy(
            model=adapted_model,
            tokenized_dataset=tokenized,
            data_collator=collator,
            sample_count=config.sample_count,
            top_k=config.top_k,
            target_modules=config.target_modules,
        )
        matched_comparison = compare_probe_rankings(
            base_modules=probe["selected_modules"],
            lora_modules=matched_lora_probe["selected_modules"],
        )
        metrics = {
            f"base_probe/module_energy/{key}": energy
            for key, energy in probe["module_energy"].items()
        }
        metrics.update(
            {
                f"base_probe/module_topk_frequency/{key}": frequency
                for key, frequency in module_frequency.items()
            }
        )
        metrics.update(
            {
                f"matched_lora_probe/module_energy/{key}": energy
                for key, energy in matched_lora_probe["module_energy"].items()
            }
        )
        run.log(metrics)
        run.summary["selected_layers"] = probe["selected_layers"]
        run.summary["selected_modules"] = probe["selected_modules"]
        run.summary["matched_lora_top_modules"] = matched_lora_probe["selected_modules"]
        run.summary["matched_top_module_overlap"] = matched_comparison["overlap"]
        run.summary["modal_artifact_path"] = config.modal_artifact_path
        result = {
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "weights_loaded": True,
            "gradient_source": "base_weights",
            "precision": "bf16",
            "optimizer_steps": 0,
            "dataset": {
                "record_count": report.record_count,
                "training_content_sha256": report.training_content_sha256,
                "holdout_content_sha256": report.holdout_content_sha256,
            },
            "probe": probe,
            "module_topk_frequency": module_frequency,
            "layer_topk_frequency": layer_frequency,
            "stability_groups": stability_groups,
            "matched_lora_probe": matched_lora_probe,
            "comparison_with_matched_lora_probe": matched_comparison,
            "comparison_with_historical_lora_probe": historical_comparison,
            "reference_run_url": reference.get("tracking", {}).get("run_url"),
            "wandb_run_url": getattr(run, "url", None),
            "modal_artifact_path": config.modal_artifact_path,
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
            "gpu_name": torch.cuda.get_device_name(0),
        }
        config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        config.metadata_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        model.zero_grad(set_to_none=True)
        run.finish()
