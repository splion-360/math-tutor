"""Serve the shared Math Tutor LoRA adapter through one pinned vLLM process.
Modal packages the verified local adapter checkpoint when deploying this module."""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "math-tutor-inference"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_REVISION = "1b4199c4f36b0cef378bfb12390c18780c18af4c"
SHARED_ADAPTER_MODEL = "shared-lora-qwen3-4b-manim-v1"
VLLM_PORT = 8000
ADAPTER_DESTINATION = "/adapter"


def resolve_adapter_source() -> Path:
    """Resolve the checkpoint during deployment and its image path at runtime."""
    configured_source = os.environ.get("MATH_TUTOR_ADAPTER_SOURCE")
    if configured_source:
        return Path(configured_source).expanduser()
    module_path = Path(__file__).resolve()
    repository_source = (
        module_path.parent.parent.parent
        / "training"
        / "artifacts"
        / "evidence"
        / "checkpoint-1790"
    )
    if repository_source.is_dir():
        return repository_source
    return Path(ADAPTER_DESTINATION)


ADAPTER_SOURCE = resolve_adapter_source()

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("math-tutor-huggingface-cache", create_if_missing=True)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(
        ADAPTER_SOURCE / "adapter_model.safetensors",
        remote_path=f"{ADAPTER_DESTINATION}/adapter_model.safetensors",
    )
    .add_local_file(
        ADAPTER_SOURCE / "adapter_config.json",
        remote_path=f"{ADAPTER_DESTINATION}/adapter_config.json",
    )
)


@app.function(
    image=vllm_image,
    gpu="L4",
    min_containers=0,
    max_containers=1,
    scaledown_window=15 * 60,
    timeout=20 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * 60, requires_proxy_auth=True)
def serve() -> None:
    """Start one OpenAI-compatible server with the shared adapter available."""
    subprocess.Popen(
        [
            "vllm",
            "serve",
            BASE_MODEL,
            "--revision",
            BASE_MODEL_REVISION,
            "--served-model-name",
            BASE_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.90",
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-lora-rank",
            "32",
            "--lora-modules",
            f"{SHARED_ADAPTER_MODEL}={ADAPTER_DESTINATION}",
        ]
    )
