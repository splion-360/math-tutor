"""Serve the pinned visual-evidence model through a scale-to-zero Modal endpoint.
The deployment is separate from lesson generation and keeps proxy credentials server-side."""

import subprocess

import modal

APP_NAME = "math-tutor-visual-validation"
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
VLLM_PORT = 8000

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("math-tutor-huggingface-cache", create_if_missing=True)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=vllm_image,
    gpu="L4",
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60, requires_proxy_auth=True)
def serve() -> None:
    """Start the pinned OpenAI-compatible visual-model server."""
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL,
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
        ]
    )
