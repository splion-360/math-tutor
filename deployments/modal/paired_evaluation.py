"""Run the frozen base-versus-shared-adapter pilot as one Modal GPU batch.
Completed responses are persisted individually; no inference endpoint stays running."""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
PLAN = ROOT / "training/artifacts/paired-pilot-v2/frozen_plan.json"
app = modal.App("math-tutor-paired-evaluation")
cache = modal.Volume.from_name("math-tutor-huggingface-cache", create_if_missing=True)
artifacts = modal.Volume.from_name(
    "math-tutor-paired-evaluation", create_if_missing=True
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "peft==0.21.0",
        "accelerate==1.10.1",
        "safetensors==0.6.2",
        "jinja2==3.1.6",
    )
    .env({"PYTHONPATH": "/workspace/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)
if modal.is_local():
    image = image.add_local_dir(ROOT / "training/src", remote_path="/workspace/src")
    image = image.add_local_file(PLAN, remote_path="/workspace/frozen_plan.json")


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    max_containers=1,
    scaledown_window=2,
    volumes={"/root/.cache/huggingface": cache, "/artifacts": artifacts},
)
def generate(run_id: str, plan_sha256: str) -> dict[str, str]:
    """Generate both conditions after verifying the uploaded frozen plan.

    Args:
        run_id: New artifact directory name.
        plan_sha256: Local preflight plan digest.

    Returns:
        Persistent output path and verified plan hash.

    Raises:
        ValueError: If the run ID or plan hash is invalid.
        RuntimeError: If generation fails; completed responses remain on the volume.
    """
    from dynamic_lora.evidence import verify_file
    from dynamic_lora.paired_generation import run_generations

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_id):
        raise ValueError("run ID must be a safe artifact name")
    plan = Path("/workspace/frozen_plan.json")
    verify_file(plan, plan_sha256)
    destination = Path("/artifacts") / run_id
    run_generations(plan_path=plan, output=destination, progress=artifacts.commit)
    return {"output": str(destination), "plan_sha256": plan_sha256}


@app.local_entrypoint()
def main(run_id: str = "paired-pilot-v2-greedy") -> None:
    """Submit the previously frozen pilot without starting a persistent service.

    Args:
        run_id: New remote run directory; existing outputs cannot be overwritten.
    """
    import hashlib

    digest = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    print(json.dumps(generate.remote(run_id, digest), indent=2))
