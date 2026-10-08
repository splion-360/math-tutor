"""Serve the base model and three local PEFT adapters through one vLLM process.
Modal packages the adapter artifacts from the repository when deploying this module."""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "math-tutor-inference"
BASE_MODEL = "Qwen/Qwen3-4B"
VLLM_PORT = 8000
REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCE = Path(
    os.environ.get(
        "MATH_TUTOR_ADAPTER_SOURCE",
        str(REPO_ROOT / "training" / "artifacts" / "token_factory"),
    )
).expanduser()
ADAPTER_DESTINATION = "/adapters"
ADAPTER_NAMES = ("foundational", "intermediate", "advanced")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("math-tutor-huggingface-cache", create_if_missing=True)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(ADAPTER_SOURCE, remote_path=ADAPTER_DESTINATION)
)


@app.function(
    image=vllm_image,
    gpu="L4",
    min_containers=1,
    max_containers=1,
    scaledown_window=15 * 60,
    timeout=20 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * 60, requires_proxy_auth=True)
def serve() -> None:
    """Start one OpenAI-compatible server with all three named LoRAs preloaded."""
    lora_modules = [
        f"{adapter}={ADAPTER_DESTINATION}/{adapter}" for adapter in ADAPTER_NAMES
    ]
    subprocess.Popen(
        [
            "vllm",
            "serve",
            BASE_MODEL,
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
            str(len(ADAPTER_NAMES)),
            "--max-lora-rank",
            "32",
            "--lora-modules",
            *lora_modules,
        ]
    )
