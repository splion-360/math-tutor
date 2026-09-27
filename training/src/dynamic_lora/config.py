"""Define and parse configuration for dynamic-LoRA training experiments.
The module validates configuration before model dependencies are loaded."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TrackingMode = Literal["auto", "disabled", "online"]


@dataclass(frozen=True)
class ExperimentTrackingConfig:
    mode: TrackingMode = "auto"
    project: str = "math-tutor-dynamic-lora"
    run_name: str | None = None
    tags: tuple[str, ...] = ("shared-lora",)
    modal_artifact_path: str | None = None


def tracking_metadata(
    config: ExperimentTrackingConfig,
    *,
    active: bool,
    git_revision: str | None,
    reason: str | None = None,
    run_url: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "wandb",
        "mode": config.mode,
        "project": config.project,
        "run_name": config.run_name,
        "tags": list(config.tags),
        "modal_artifact_path": config.modal_artifact_path,
        "active": active,
        "git_revision": git_revision,
    }
    if reason is not None:
        metadata["reason"] = reason
    if run_url is not None:
        metadata["run_url"] = run_url
    return metadata


@dataclass(frozen=True)
class TrainingConfig:
    train_path: Path
    holdout_path: Path
    output_dir: Path
    metadata_path: Path
    seed: int = 42
    max_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    )
    load_in_4bit: bool = True
    run_smoke_eval: bool = False
    layer_energy_probe_top_k: int = 0
    layer_energy_probe_sample_count: int = 16
    gradient_signature_dim: int = 0
    gradient_signature_every_steps: int = 1
    gradient_signature_start_step: int = 1
    signature_probe_metadata_path: Path | None = None
    fixed_prompt_probe_sample_count: int = 0
    fixed_prompt_probe_steps: tuple[int, ...] = ()
    tracking: ExperimentTrackingConfig = field(default_factory=ExperimentTrackingConfig)


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> TrainingConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    base_dir = path.resolve().parent
    return parse_config(raw, base_dir=base_dir)


def parse_config(raw: dict[str, Any], *, base_dir: Path) -> TrainingConfig:
    forbidden = {
        "model_id",
        "condition",
        "adapter_id",
        "dynamic_adapter_spawning",
        "learned_router",
    }
    present_forbidden = sorted(forbidden & raw.keys())
    if present_forbidden:
        joined = ", ".join(present_forbidden)
        raise ConfigError(f"frozen baseline fields are not configurable: {joined}")

    def path_field(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{name} must be a non-empty string")
        path = Path(value)
        return path if path.is_absolute() else base_dir / path

    def int_field(name: str, default: int) -> int:
        value = raw.get(name, default)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer")
        return value

    def non_negative_int_field(name: str, default: int) -> int:
        value = raw.get(name, default)
        if not isinstance(value, int) or value < 0:
            raise ConfigError(f"{name} must be a non-negative integer")
        return value

    def float_field(name: str, default: float) -> float:
        value = raw.get(name, default)
        if not isinstance(value, int | float) or value <= 0:
            raise ConfigError(f"{name} must be a positive number")
        return float(value)

    target_modules_raw = raw.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if (
        not isinstance(target_modules_raw, list)
        or not target_modules_raw
        or not all(isinstance(item, str) and item for item in target_modules_raw)
    ):
        raise ConfigError("target_modules must be a non-empty list of strings")

    load_in_4bit = raw.get("load_in_4bit", True)
    if not isinstance(load_in_4bit, bool):
        raise ConfigError("load_in_4bit must be a boolean")

    run_smoke_eval = raw.get("run_smoke_eval", False)
    if not isinstance(run_smoke_eval, bool):
        raise ConfigError("run_smoke_eval must be a boolean")

    lora_dropout = raw.get("lora_dropout", 0.05)
    if not isinstance(lora_dropout, int | float) or not 0 <= lora_dropout < 1:
        raise ConfigError("lora_dropout must be in [0, 1)")

    layer_energy_probe_top_k = non_negative_int_field("layer_energy_probe_top_k", 0)
    gradient_signature_dim = non_negative_int_field("gradient_signature_dim", 0)
    signature_probe_metadata_path = (
        path_field("signature_probe_metadata_path")
        if "signature_probe_metadata_path" in raw
        else None
    )
    max_steps = int_field("max_steps", 1)
    gradient_signature_start_step = int_field("gradient_signature_start_step", 1)
    if gradient_signature_dim > 0 and (
        (layer_energy_probe_top_k > 0) == (signature_probe_metadata_path is not None)
    ):
        raise ConfigError(
            "gradient_signature_dim requires exactly one of layer_energy_probe_top_k "
            "or signature_probe_metadata_path"
        )
    if gradient_signature_dim > 0 and gradient_signature_start_step > max_steps:
        raise ConfigError("gradient_signature_start_step cannot exceed max_steps")

    fixed_prompt_probe_sample_count = non_negative_int_field(
        "fixed_prompt_probe_sample_count", 0
    )
    fixed_prompt_probe_steps = raw.get("fixed_prompt_probe_steps", [])
    if (
        not isinstance(fixed_prompt_probe_steps, list)
        or any(
            type(step) is not int or step < 0 or step > max_steps
            for step in fixed_prompt_probe_steps
        )
        or fixed_prompt_probe_steps != sorted(set(fixed_prompt_probe_steps))
        or (fixed_prompt_probe_sample_count > 0) != bool(fixed_prompt_probe_steps)
        or (fixed_prompt_probe_sample_count > 0 and gradient_signature_dim == 0)
    ):
        raise ConfigError(
            "fixed_prompt_probe_steps must be unique ordered steps within training, "
            "with a positive sample count and gradient signatures enabled"
        )

    return TrainingConfig(
        train_path=path_field("train_path"),
        holdout_path=path_field("holdout_path"),
        output_dir=path_field("output_dir"),
        metadata_path=path_field("metadata_path"),
        seed=int_field("seed", 42),
        max_steps=max_steps,
        per_device_train_batch_size=int_field("per_device_train_batch_size", 1),
        gradient_accumulation_steps=int_field("gradient_accumulation_steps", 4),
        learning_rate=float_field("learning_rate", 2e-4),
        max_seq_length=int_field("max_seq_length", 2048),
        lora_r=int_field("lora_r", 16),
        lora_alpha=int_field("lora_alpha", 32),
        lora_dropout=float(lora_dropout),
        target_modules=tuple(target_modules_raw),
        load_in_4bit=load_in_4bit,
        run_smoke_eval=run_smoke_eval,
        layer_energy_probe_top_k=layer_energy_probe_top_k,
        layer_energy_probe_sample_count=int_field("layer_energy_probe_sample_count", 16),
        gradient_signature_dim=gradient_signature_dim,
        gradient_signature_every_steps=int_field("gradient_signature_every_steps", 1),
        gradient_signature_start_step=gradient_signature_start_step,
        signature_probe_metadata_path=signature_probe_metadata_path,
        fixed_prompt_probe_sample_count=fixed_prompt_probe_sample_count,
        fixed_prompt_probe_steps=tuple(fixed_prompt_probe_steps),
        tracking=_tracking_config(raw.get("tracking")),
    )


def _tracking_config(raw: object) -> ExperimentTrackingConfig:
    if raw is None:
        return ExperimentTrackingConfig()
    if not isinstance(raw, dict):
        raise ConfigError("tracking must be an object")

    mode = raw.get("mode", "auto")
    if mode not in ("auto", "disabled", "online"):
        raise ConfigError("tracking.mode must be auto, disabled, or online")

    project = raw.get("project", "math-tutor-dynamic-lora")
    if not isinstance(project, str) or not project:
        raise ConfigError("tracking.project must be a non-empty string")

    run_name = raw.get("run_name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name):
        raise ConfigError("tracking.run_name must be a non-empty string when set")

    tags_raw = raw.get("tags", ["shared-lora"])
    if (
        not isinstance(tags_raw, list)
        or not all(isinstance(item, str) and item for item in tags_raw)
    ):
        raise ConfigError("tracking.tags must be a list of non-empty strings")

    modal_artifact_path = raw.get("modal_artifact_path")
    if modal_artifact_path is not None and (
        not isinstance(modal_artifact_path, str) or not modal_artifact_path
    ):
        raise ConfigError("tracking.modal_artifact_path must be a non-empty string when set")

    return ExperimentTrackingConfig(
        mode=mode,
        project=project,
        run_name=run_name,
        tags=tuple(tags_raw),
        modal_artifact_path=modal_artifact_path,
    )
