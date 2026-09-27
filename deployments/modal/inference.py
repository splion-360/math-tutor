"""Serve Qwen3-4B and the three local PEFT adapters through one vLLM process.

Run this from the repository root with `modal deploy deployments/modal/inference.py`.
The adapter source is deliberately outside this worktree and is mounted read-only at deploy time;
no weight files are copied into or committed from this repository.
"""

import subprocess
from pathlib import Path

import modal

APP_NAME = "dream-ai-qwen3-lora"
BASE_MODEL = "Qwen/Qwen3-4B"
VLLM_PORT = 8000
ADAPTER_SOURCE = Path(
    "/Users/bladeofchaos/Desktop/personal/hackathon/dream-ai-hackathon/"
    "training/artifacts/token_factory"
)
ADAPTER_DESTINATION = "/adapters"
ADAPTER_NAMES = ("foundational", "intermediate", "advanced")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("dream-ai-huggingface-cache", create_if_missing=True)
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
