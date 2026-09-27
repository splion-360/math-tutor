"""Run the shared-LoRA training smoke job on Modal.

Run this from the repository root with:

    modal run modal/shared_lora_training_smoke.py

The job uses the tiny checked-in training fixture. It is a wiring smoke test, not a quality run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dream-ai-shared-lora-training-smoke"
HF_CACHE_VOLUME = "dream-ai-huggingface-cache"
TRAINING_ARTIFACT_VOLUME = "dream-ai-training-artifacts"
WANDB_SECRET_NAME = "wandb-api-key"

REPO_ROOT = Path(__file__).parents[1]
TRAINING_SOURCE = REPO_ROOT / "training"
TRAINING_SRC_SOURCE = TRAINING_SOURCE / "src"
TRAINING_FIXTURES_SOURCE = TRAINING_SOURCE / "fixtures"
TRAINING_CONFIGS_SOURCE = TRAINING_SOURCE / "configs"
HOLDOUT_SOURCE = REPO_ROOT / "backend" / "data" / "evaluation"
REMOTE_ROOT = Path("/workspace")
REMOTE_TRAINING_ROOT = REMOTE_ROOT / "training"
REMOTE_HOLDOUT_ROOT = REMOTE_ROOT / "backend" / "data" / "evaluation"
REMOTE_CONFIG_PATH = REMOTE_TRAINING_ROOT / "configs" / "modal_smoke_qwen3_4b.json"
REMOTE_ARTIFACT_ROOT = Path("/artifacts")


def _train_requirements() -> tuple[str, ...]:
    pyproject = tomllib.loads((TRAINING_SOURCE / "pyproject.toml").read_text(encoding="utf-8"))
    return tuple(pyproject["project"]["optional-dependencies"]["train"])


def _source_version() -> str:
    env_revision = os.environ.get("GIT_REVISION") or os.environ.get("GITHUB_SHA")
    if env_revision:
        return env_revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
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
def run_smoke() -> dict[str, Any]:
    """Train and evaluate one tiny shared-LoRA smoke run."""
    from shared_lora_baseline.config import load_config
    from shared_lora_baseline.trainer import train_shared_lora

    config = load_config(REMOTE_CONFIG_PATH)
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
        "trainable_parameters": metadata["parameter_budget"]["trainable_parameters"],
        "plan": plan.redacted_text,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
