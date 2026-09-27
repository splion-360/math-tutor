"""Train and evaluate one fixed LoRA placement on Modal.
The run keeps datasets, optimizer settings, and generation prompts equal across arms."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import (
    format_training_prompt,
    load_training_records,
    tokenize_completion_record,
    validate_training_dataset,
)
from dynamic_lora.placement import (
    assert_exact_trainable_modules,
    build_placement_manifest,
    peft_target_modules,
)


@dataclass(frozen=True)
class PlacementConfig:
    """Frozen inputs and training settings for the four placement arms."""

    train_path: Path
    holdout_path: Path
    probe_metadata_path: Path
    artifact_root: Path
    seed: int
    max_steps: int
    max_seq_length: int
    max_new_tokens: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    wandb_project: str


def load_placement_config(path: Path) -> PlacementConfig:
    """Load a placement experiment config with bounded numeric settings.

    Args:
        path: JSON config location; relative paths resolve beside it.

    Returns:
        Validated immutable experiment configuration.

    Raises:
        ValueError: If required fields have invalid types or ranges.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("placement config must be an object")

    def location(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty path")
        selected = Path(value)
        return selected if selected.is_absolute() else path.parent / selected

    def positive_int(name: str) -> int:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def positive_float(name: str) -> float:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return float(value)

    dropout = raw.get("lora_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, int | float) or not 0 <= dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    project = raw.get("wandb_project")
    if not isinstance(project, str) or not project:
        raise ValueError("wandb_project must be a non-empty string")
    max_seq_length = positive_int("max_seq_length")
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    return PlacementConfig(
        train_path=location("train_path"),
        holdout_path=location("holdout_path"),
        probe_metadata_path=location("probe_metadata_path"),
        artifact_root=location("artifact_root"),
        seed=positive_int("seed"),
        max_steps=positive_int("max_steps"),
        max_seq_length=max_seq_length,
        max_new_tokens=positive_int("max_new_tokens"),
        lora_r=positive_int("lora_r"),
        lora_alpha=positive_int("lora_alpha"),
        lora_dropout=float(dropout),
        learning_rate=positive_float("learning_rate"),
        wandb_project=project,
    )


def validate_placement_probe(
    metadata: dict[str, Any], *, train_hash: str, holdout_hash: str
) -> None:
    """Require a no-update base probe on the exact training and holdout files.

    Args:
        metadata: Saved base-gradient probe metadata.
        train_hash: SHA-256 of the current training JSONL.
        holdout_hash: SHA-256 of the current prompt-only evaluation JSONL.

    Raises:
        ValueError: If model, gradient source, or data provenance differs.
    """
    if (metadata.get("model_id"), metadata.get("model_revision")) != (
        FROZEN_MODEL_ID,
        FROZEN_MODEL_REVISION,
    ):
        raise ValueError("placement probe uses a different model revision")
    if metadata.get("gradient_source") != "base_weights" or metadata.get("optimizer_steps") != 0:
        raise ValueError("placement probe must use untrained base-weight gradients")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("training_content_sha256") != train_hash:
        raise ValueError("placement probe training dataset does not match")
    if dataset.get("holdout_content_sha256") != holdout_hash:
        raise ValueError("placement probe holdout dataset does not match")


def run_placement_arm(config: PlacementConfig, arm_name: str) -> dict[str, Any]:
    """Train one exact four-module arm and record loss plus direct code generations.

    Args:
        config: Frozen placement experiment settings.
        arm_name: One of discovered, low_energy, random_1, or random_2.

    Returns:
        Run metadata including train/validation metrics and output locations.

    Raises:
        ValueError: If inputs differ from the probe or the arm is unknown.
        RuntimeError: If W&B is unavailable or generated model setup is invalid.
    """
    report = validate_training_dataset(config.train_path, config.holdout_path)
    probe = json.loads(config.probe_metadata_path.read_text(encoding="utf-8"))
    validate_placement_probe(
        probe,
        train_hash=report.training_content_sha256,
        holdout_hash=report.holdout_content_sha256,
    )
    records = load_training_records(config.train_path)
    manifest = build_placement_manifest(probe, records, split_seed=config.seed)
    if arm_name not in manifest["arms"]:
        raise ValueError(f"unknown placement arm: {arm_name}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for placement experiment tracking")
    selected = manifest["arms"][arm_name]
    run_dir = config.artifact_root / arm_name
    if (run_dir / "run_metadata.json").exists():
        raise ValueError(f"placement arm already completed: {arm_name}")

    datasets = import_module("datasets")
    peft = import_module("peft")
    torch = import_module("torch")
    transformers = import_module("transformers")
    wandb = import_module("wandb")
    transformers.set_seed(config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FROZEN_MODEL_ID, revision=FROZEN_MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training_ids = set(manifest["training_ids"])
    validation_ids = set(manifest["validation_ids"])
    train_dataset = datasets.Dataset.from_list(
        [
            tokenize_completion_record(record, tokenizer, max_seq_length=config.max_seq_length)
            for record in records
            if record["id"] in training_ids
        ]
    )
    validation_dataset = datasets.Dataset.from_list(
        [
            tokenize_completion_record(record, tokenizer, max_seq_length=config.max_seq_length)
            for record in records
            if record["id"] in validation_ids
        ]
    )
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        FROZEN_MODEL_ID,
        revision=FROZEN_MODEL_REVISION,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = peft.prepare_model_for_kbit_training(model)
    adapter_config = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=peft_target_modules(selected),
        task_type=peft.TaskType.CAUSAL_LM,
    )
    model = peft.get_peft_model(model, adapter_config)
    trainable_tensors = assert_exact_trainable_modules(model.named_parameters(), selected)
    trainable_parameters, _ = model.get_nb_trainable_parameters()

    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=f"placement-{arm_name}-{config.max_steps}",
        job_type="placement-ablation",
        config={
            "arm": arm_name,
            "selected_modules": selected,
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "training_content_sha256": report.training_content_sha256,
            "holdout_content_sha256": report.holdout_content_sha256,
            "seed": config.seed,
            "max_steps": config.max_steps,
            "max_seq_length": config.max_seq_length,
            "max_new_tokens": config.max_new_tokens,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "trainable_parameters": trainable_parameters,
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
        },
    )
    try:
        training_args = transformers.TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            seed=config.seed,
            data_seed=config.seed,
            max_steps=config.max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=config.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to="wandb",
        )
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer, label_pad_token_id=-100
            ),
        )
        train_result = trainer.train()
        validation_metrics = trainer.evaluate()
        model.save_pretrained(run_dir / "adapter")
        tokenizer.save_pretrained(run_dir / "adapter")
        model.config.use_cache = True
        model.eval()
        evaluation_records = [
            json.loads(line)
            for line in config.holdout_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        generations: list[dict[str, Any]] = []
        for record in evaluation_records:
            prompt = format_training_prompt(record, tokenizer)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
                model.device
            )
            started = monotonic()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            completion = output[0][inputs["input_ids"].shape[-1] :]
            generations.append(
                {
                    "id": record["id"],
                    "difficulty": record["difficulty"],
                    "topic": record["topic"],
                    "prompt": record["prompt"],
                    "response": tokenizer.decode(completion, skip_special_tokens=True),
                    "completion_tokens": len(completion),
                    "hit_token_limit": len(completion) >= config.max_new_tokens,
                    "generation_latency_seconds": monotonic() - started,
                }
            )
            (run_dir / "generations.jsonl").write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in generations),
                encoding="utf-8",
            )
        wandb_run.log(
            {
                "placement/validation_code_loss": validation_metrics["eval_loss"],
                "placement/train_loss": train_result.metrics.get("train_loss"),
                "placement/generation_count": len(generations),
                "placement/token_limit_count": sum(item["hit_token_limit"] for item in generations),
            }
        )
        result = {
            "arm": arm_name,
            "selected_modules": selected,
            "manifest": manifest,
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "probe_run_url": probe.get("wandb_run_url"),
            "dataset": {
                "training_content_sha256": report.training_content_sha256,
                "holdout_content_sha256": report.holdout_content_sha256,
                "training_count": len(train_dataset),
                "validation_count": len(validation_dataset),
                "generation_count": len(generations),
            },
            "trainable_tensors": trainable_tensors,
            "trainable_parameters": trainable_parameters,
            "training_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
            "max_steps": config.max_steps,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
            "wandb_run_url": getattr(wandb_run, "url", None),
            "adapter_path": str(run_dir / "adapter"),
            "generations_path": str(run_dir / "generations.jsonl"),
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        wandb_run.finish()
