"""Run shared-LoRA training and base-weight probe jobs on Modal.

Run this from the repository root with:

    modal run deployments/modal/training_smoke.py

The default job uses the tiny checked-in fixture. Full-dataset modes provide
LoRA-gradient diagnostics, signatures, or a separate base-weight probe.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal
import tomllib

APP_NAME = "dream-ai-shared-lora-training-smoke"
HF_CACHE_VOLUME = "dream-ai-huggingface-cache"
TRAINING_ARTIFACT_VOLUME = "dream-ai-training-artifacts"
WANDB_SECRET_NAME = "WANDB_API_KEY"
TRAIN_REQUIREMENTS_FALLBACK = (
    "accelerate>=1.1,<2",
    "bitsandbytes>=0.43,<1",
    "datasets>=2.21,<3",
    "peft>=0.12,<1",
    "safetensors>=0.4,<1",
    "torch>=2.4,<3",
    "transformers>=4.51,<5",
    "wandb>=0.18,<1",
)

MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = MODULE_PATH.parents[2] if len(MODULE_PATH.parents) > 2 else Path("/workspace")
TRAINING_SOURCE = REPO_ROOT / "training"
TRAINING_SRC_SOURCE = TRAINING_SOURCE / "src"
TRAINING_FIXTURES_SOURCE = TRAINING_SOURCE / "fixtures"
TRAINING_CONFIGS_SOURCE = TRAINING_SOURCE / "configs"
FULL_TRAINING_DATA_SOURCE = TRAINING_SOURCE / "data" / "bespoke_manim_train.jsonl"
HOLDOUT_SOURCE = REPO_ROOT / "backend" / "data" / "evaluation"
REMOTE_ROOT = Path("/workspace")
REMOTE_TRAINING_ROOT = REMOTE_ROOT / "training"
REMOTE_HOLDOUT_ROOT = REMOTE_ROOT / "backend" / "data" / "evaluation"
REMOTE_CONFIG_DIR = REMOTE_TRAINING_ROOT / "configs"
REMOTE_FULL_TRAINING_DATA = REMOTE_TRAINING_ROOT / "data" / "bespoke_manim_train.jsonl"
REMOTE_ARTIFACT_ROOT = Path("/artifacts")


def _train_requirements() -> tuple[str, ...]:
    pyproject_path = TRAINING_SOURCE / "pyproject.toml"
    if not pyproject_path.exists():
        return TRAIN_REQUIREMENTS_FALLBACK
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return tuple(pyproject["project"]["optional-dependencies"]["train"])


def _source_version() -> str:
    env_revision = os.environ.get("GIT_REVISION") or os.environ.get("GITHUB_SHA")
    if env_revision:
        return env_revision
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
training_artifacts = modal.Volume.from_name(TRAINING_ARTIFACT_VOLUME, create_if_missing=True)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(*_train_requirements())
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": str(REMOTE_TRAINING_ROOT / "src"),
            "SOURCE_VERSION": _source_version(),
        }
    )
    .add_local_dir(TRAINING_SRC_SOURCE, remote_path=str(REMOTE_TRAINING_ROOT / "src"))
    .add_local_dir(TRAINING_FIXTURES_SOURCE, remote_path=str(REMOTE_TRAINING_ROOT / "fixtures"))
    .add_local_dir(TRAINING_CONFIGS_SOURCE, remote_path=str(REMOTE_TRAINING_ROOT / "configs"))
    .add_local_dir(HOLDOUT_SOURCE, remote_path=str(REMOTE_HOLDOUT_ROOT))
)
if FULL_TRAINING_DATA_SOURCE.exists():
    train_image = train_image.add_local_file(
        FULL_TRAINING_DATA_SOURCE, remote_path=str(REMOTE_FULL_TRAINING_DATA)
    )


@app.function(
    image=train_image,
    gpu="L4",
    timeout=45 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        str(REMOTE_ARTIFACT_ROOT): training_artifacts,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_smoke(config_name: str = "modal_smoke_qwen3_4b.json") -> dict[str, Any]:
    """Run a shared-LoRA smoke or layer-probe configuration on Modal.

    Args:
        config_name: Name of a mounted JSON config in the training config directory.

    Returns:
        Run identity, diagnostic summary, and artifact locations.
    """
    return _run_shared_training(config_name)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=45 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        str(REMOTE_ARTIFACT_ROOT): training_artifacts,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_full_probe(config_name: str) -> dict[str, Any]:
    """Run a full-dataset LoRA probe on a GPU sized for complete examples.

    Args:
        config_name: Name of a mounted full-dataset JSON config.

    Returns:
        Run identity, probe summary, and artifact locations.
    """
    return _run_shared_training(config_name)


def _run_shared_training(config_name: str) -> dict[str, Any]:
    from dynamic_lora.config import load_config
    from dynamic_lora.training import train_shared_lora

    config = load_config(REMOTE_CONFIG_DIR / config_name)
    plan = train_shared_lora(config)
    training_artifacts.commit()
    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    return {
        "adapter_id": metadata["adapter_id"],
        "condition": metadata["condition"],
        "metadata_path": str(config.metadata_path),
        "modal_artifact_path": metadata["tracking"]["modal_artifact_path"],
        "tracking_active": metadata["tracking"]["active"],
        "wandb_run_url": metadata["tracking"].get("run_url"),
        "weights_loaded": metadata["weights_loaded"],
        "smoke_eval": metadata.get("smoke_eval"),
        "layer_energy_probe": metadata.get("layer_energy_probe"),
        "gradient_signatures": metadata.get("gradient_signatures"),
        "trainable_parameters": metadata["parameter_budget"]["trainable_parameters"],
        "plan": plan.redacted_text,
    }


@app.function(
    image=train_image,
    gpu="A100-40GB",
    timeout=45 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        str(REMOTE_ARTIFACT_ROOT): training_artifacts,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_base_weight_probe() -> dict[str, Any]:
    """Measure BF16 base-weight gradients without training or attaching LoRA.

    Returns:
        Ranked layers and modules, overlap with the prior LoRA probe, and run URLs.
    """
    from dynamic_lora.base_probe_run import load_base_probe_config, run_base_probe

    config = load_base_probe_config(REMOTE_CONFIG_DIR / "modal_base_weight_probe_qwen3_4b.json")
    result = run_base_probe(config)
    training_artifacts.commit()
    return {
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "gradient_source": result["gradient_source"],
        "optimizer_steps": result["optimizer_steps"],
        "selected_layers": result["probe"]["selected_layers"],
        "selected_modules": result["probe"]["selected_modules"],
        "matched_lora_modules": result["matched_lora_probe"]["selected_modules"],
        "comparison_with_matched_lora_probe": result[
            "comparison_with_matched_lora_probe"
        ],
        "comparison_with_historical_lora_probe": result[
            "comparison_with_historical_lora_probe"
        ],
        "wandb_run_url": result["wandb_run_url"],
        "modal_artifact_path": result["modal_artifact_path"],
        "metadata_path": str(config.metadata_path),
    }


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=90 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        str(REMOTE_ARTIFACT_ROOT): training_artifacts,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_placement_arm(arm_name: str) -> dict[str, Any]:
    """Train and validate one preselected four-module placement arm.

    Args:
        arm_name: Discovered, low_energy, random_1, or random_2.

    Returns:
        Run identity, validation loss, and adapter artifact path.
    """
    from dynamic_lora.placement_run import load_placement_config, run_placement_arm as run_arm

    config = load_placement_config(REMOTE_CONFIG_DIR / "modal_placement_ablation_qwen3_4b.json")
    result = run_arm(config, arm_name)
    training_artifacts.commit()
    return {
        "arm": result["arm"],
        "selected_modules": result["selected_modules"],
        "trainable_parameters": result["trainable_parameters"],
        "validation_code_loss": result["validation_metrics"]["eval_loss"],
        "adapter_path": result["adapter_path"],
        "wandb_run_url": result["wandb_run_url"],
        "metadata_path": str(config.artifact_root / arm_name / "run_metadata.json"),
    }


@app.local_entrypoint()
def main(
    full_probe: bool = False,
    signatures: bool = False,
    base_probe: bool = False,
    placement_arm: str = "",
) -> None:
    """Choose the tiny fixture or one full-dataset diagnostic run."""
    if sum((full_probe, signatures, base_probe, bool(placement_arm))) > 1:
        raise ValueError("choose only one full-dataset diagnostic mode")
    if (full_probe or signatures or base_probe or placement_arm) and not FULL_TRAINING_DATA_SOURCE.exists():
        raise FileNotFoundError(FULL_TRAINING_DATA_SOURCE)
    if placement_arm:
        if placement_arm not in {"discovered", "low_energy", "random_1", "random_2"}:
            raise ValueError(f"unknown placement arm: {placement_arm}")
        print(json.dumps(run_placement_arm.remote(placement_arm), indent=2, sort_keys=True))
        return
    if base_probe:
        print(json.dumps(run_base_weight_probe.remote(), indent=2, sort_keys=True))
        return
    if signatures:
        config_name = "modal_signatures_qwen3_4b.json"
    elif full_probe:
        config_name = "modal_probe_qwen3_4b.json"
    else:
        config_name = "modal_smoke_qwen3_4b.json"
    run_function = run_full_probe if full_probe or signatures else run_smoke
    result = run_function.remote(config_name)
    print(json.dumps(result, indent=2, sort_keys=True))
