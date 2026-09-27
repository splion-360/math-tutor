"""Train a shared LoRA across complete Manim epochs and inspect row gradients.
The experiment keeps its supervised split and artifacts separate from short probes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import (
    load_training_records,
    tokenize_completion_record,
    validate_training_dataset,
)
from dynamic_lora.fixed_prompt_signatures import (
    ProbeExample,
    build_fixed_prompt_signature_callback,
)
from dynamic_lora.full_corpus_plan import FullCorpusPlan, build_full_corpus_plan
from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key

PROJECTION_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class FullCorpusConfig:
    """Frozen inputs for the full-corpus gradient-direction experiment."""

    train_path: Path
    holdout_path: Path
    artifact_root: Path
    seed: int
    epochs: int
    max_seq_length: int
    learning_rate: float
    projection_dim: int
    expected_training_count: int | None = None
    expected_validation_count: int | None = None


def load_full_corpus_config(path: Path) -> FullCorpusConfig:
    """Load paths and bounded numeric settings before model dependencies.

    Args:
        path: JSON experiment configuration path.

    Returns:
        Validated, immutable run settings.

    Raises:
        ValueError: If required configuration is absent or invalid.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("full-corpus config must be an object")

    def location(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty path")
        selected = Path(value)
        return selected if selected.is_absolute() else path.parent / selected

    def positive_int(name: str) -> int:
        value = raw.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    learning_rate = raw.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, int | float)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive")
    return FullCorpusConfig(
        train_path=location("train_path"),
        holdout_path=location("holdout_path"),
        artifact_root=location("artifact_root"),
        seed=positive_int("seed"),
        epochs=positive_int("epochs"),
        max_seq_length=positive_int("max_seq_length"),
        learning_rate=float(learning_rate),
        projection_dim=positive_int("projection_dim"),
        expected_training_count=(
            positive_int("expected_training_count") if "expected_training_count" in raw else None
        ),
        expected_validation_count=(
            positive_int("expected_validation_count")
            if "expected_validation_count" in raw
            else None
        ),
    )


def build_tokenized_splits(
    records: list[dict[str, Any]], tokenizer: Any, plan: FullCorpusPlan, *, max_seq_length: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tokenize each source row once with code-only supervised labels.

    Args:
        records: Validated Manim source records.
        tokenizer: Frozen Qwen tokenizer.
        plan: Disjoint training and validation IDs.
        max_seq_length: Complete prompt-plus-code token ceiling.

    Returns:
        Identified tokenized rows for training and validation, respectively.
    """
    training_ids = set(plan.training_ids)
    validation_ids = set(plan.validation_ids)
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        row = {
            "id": record["id"],
            "tokenized": tokenize_completion_record(
                record, tokenizer, max_seq_length=max_seq_length
            ),
        }
        if record["id"] in training_ids:
            training.append(row)
        elif record["id"] in validation_ids:
            validation.append(row)
        else:
            raise ValueError(f"record {record['id']} is absent from the corpus plan")
    if len(training) != len(training_ids) or len(validation) != len(validation_ids):
        raise ValueError("corpus plan does not match the tokenized dataset")
    return training, validation


def training_arguments_kwargs(
    *, output_dir: Path, seed: int, epochs: int, learning_rate: float = 2e-4
) -> dict[str, Any]:
    """Build epoch-based training arguments with held-out evaluation.

    Args:
        output_dir: Checkpoint directory for this experiment.
        seed: Trainer and data-shuffling seed.
        epochs: Number of complete passes over the training rows.
        learning_rate: LoRA optimizer learning rate.

    Returns:
        Keyword arguments for Transformers TrainingArguments.
    """
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    return {
        "output_dir": str(output_dir),
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 25,
        "report_to": "wandb",
    }


def choose_monitor_modules(parameter_names: list[str], *, layer_count: int) -> tuple[str, ...]:
    """Select every projection category at evenly spaced transformer depths.

    Args:
        parameter_names: Names of actual trainable LoRA tensors on the model.
        layer_count: Number of transformer depths to monitor.

    Returns:
        Stable layer-module keys spanning the observed model depth.

    Raises:
        ValueError: If requested depths or categories are unavailable.
    """
    keys = {lora_layer_key(name) for name in parameter_names if is_lora_parameter(name)}
    depths = sorted(
        {int(key.split(".")[0].removeprefix("layer_")) for key in keys if key.startswith("layer_")}
    )
    if layer_count < 2 or layer_count > len(depths):
        raise ValueError("monitor layer_count must fit the model and be at least two")
    positions = {
        round(index * (len(depths) - 1) / (layer_count - 1)) for index in range(layer_count)
    }
    if len(positions) != layer_count:
        raise ValueError("monitor layer positions are not distinct")
    selected = tuple(
        f"layer_{depths[position]}.{category}"
        for position in sorted(positions)
        for category in PROJECTION_CATEGORIES
    )
    if not set(selected).issubset(keys):
        raise ValueError("one or more monitoring projections are missing from the model")
    return selected


def run_full_corpus(config: FullCorpusConfig) -> dict[str, Any]:
    """Train complete epochs and probe every training row at the best checkpoint.

    Args:
        config: Frozen experiment paths and training settings.

    Returns:
        Saved run metadata, including exact row IDs and gradient artifact paths.

    Raises:
        ValueError: If data or observed training coverage differs from the plan.
        RuntimeError: If W&B credentials or monitoring modules are unavailable.
    """
    report = validate_training_dataset(config.train_path, config.holdout_path)
    records = load_training_records(config.train_path)
    plan = build_full_corpus_plan(records, seed=config.seed, epochs=config.epochs)
    if (
        config.expected_training_count is not None
        and len(plan.training_ids) != config.expected_training_count
    ):
        raise ValueError("training row count differs from the approved full-corpus plan")
    if (
        config.expected_validation_count is not None
        and len(plan.validation_ids) != config.expected_validation_count
    ):
        raise ValueError("validation row count differs from the approved full-corpus plan")
    metadata_path = config.artifact_root / "run_metadata.json"
    if metadata_path.exists():
        raise ValueError(f"full-corpus run already completed: {metadata_path}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for full-corpus tracking")

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
    training_rows, validation_rows = build_tokenized_splits(
        records, tokenizer, plan, max_seq_length=config.max_seq_length
    )
    train_dataset = datasets.Dataset.from_list([row["tokenized"] for row in training_rows])
    validation_dataset = datasets.Dataset.from_list([row["tokenized"] for row in validation_rows])

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
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=list(PROJECTION_CATEGORIES),
            task_type=peft.TaskType.CAUSAL_LM,
        ),
    )
    monitored_modules = choose_monitor_modules(
        [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        layer_count=4,
    )
    trainable_parameters, _ = model.get_nb_trainable_parameters()

    config.artifact_root.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project="math-tutor-dynamic-lora",
        name="full-corpus-shared-lora-3-epochs",
        job_type="full-corpus-gradients",
        config={
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "train_sha256": report.training_content_sha256,
            "seed": config.seed,
            "epochs": config.epochs,
            "training_count": len(training_rows),
            "validation_count": len(validation_rows),
            "loss_target": "assistant_code_only",
            "monitor_modules": list(monitored_modules),
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
        },
    )
    try:
        training_args = transformers.TrainingArguments(
            **training_arguments_kwargs(
                output_dir=config.artifact_root / "checkpoints",
                seed=config.seed,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
            )
        )
        collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
        )
        train_result = trainer.train()
        if train_result.global_step != plan.total_training_presentations:
            raise ValueError(
                "full-corpus training stopped before every planned row was presented: "
                f"{train_result.global_step} steps vs {plan.total_training_presentations} expected"
            )
        validation_metrics = trainer.evaluate()
        best_checkpoint = trainer.state.best_model_checkpoint
        if not isinstance(best_checkpoint, str) or not Path(best_checkpoint).name.startswith(
            "checkpoint-"
        ):
            raise ValueError("Trainer did not report a best saved checkpoint")
        probe_step = int(Path(best_checkpoint).name.removeprefix("checkpoint-"))
        if not 0 < probe_step <= train_result.global_step:
            raise ValueError("best checkpoint step is outside the completed training run")
        probe = build_fixed_prompt_signature_callback(
            selected_layers=monitored_modules,
            examples=tuple(ProbeExample(row["id"], row["tokenized"]) for row in training_rows),
            steps=(probe_step,),
            projection_dim=config.projection_dim,
            seed=config.seed,
            artifact_dir=config.artifact_root / "per_example_gradients",
            data_collator=collator,
            torch_module=torch,
            callback_base=transformers.TrainerCallback,
        )
        probe.capture(step=probe_step, model=model)
        gradient_probe = probe.finalize()
        model.save_pretrained(config.artifact_root / "adapter")
        tokenizer.save_pretrained(config.artifact_root / "adapter")
        result = {
            "model_id": FROZEN_MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "config": {
                **asdict(config),
                "train_path": str(config.train_path),
                "holdout_path": str(config.holdout_path),
                "artifact_root": str(config.artifact_root),
            },
            "dataset": {
                "source_count": report.record_count,
                "training_count": len(training_rows),
                "validation_count": len(validation_rows),
                "training_ids": list(plan.training_ids),
                "validation_ids": list(plan.validation_ids),
                "training_content_sha256": report.training_content_sha256,
                "holdout_content_sha256": report.holdout_content_sha256,
            },
            "training": {
                "epochs_planned": config.epochs,
                "examples_per_epoch": plan.examples_per_epoch,
                "presentations_planned": plan.total_training_presentations,
                "optimizer_steps_observed": train_result.global_step,
                "best_model_checkpoint": best_checkpoint,
                "probe_checkpoint_step": probe_step,
                "loss_target": "assistant_code_only",
                "metrics": train_result.metrics,
            },
            "validation_metrics": validation_metrics,
            "monitor_modules": list(monitored_modules),
            "gradient_probe": gradient_probe,
            "trainable_parameters": trainable_parameters,
            "wandb_run_url": getattr(wandb_run, "url", None),
            "git_revision": os.environ.get("SOURCE_VERSION", "unknown"),
        }
        metadata_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        wandb_run.finish()
