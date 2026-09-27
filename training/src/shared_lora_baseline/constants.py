from __future__ import annotations

FROZEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
FROZEN_MODEL_REVISION = "1b4199c4f36b0cef378bfb12390c18780c18af4c"
CONDITION = "shared_lora_static_control"
ADAPTER_ID = "shared-lora-qwen3-4b-manim-v1"
ADAPTER_KIND = "static_shared_lora"
ALLOWED_DIFFICULTIES = ("foundational", "intermediate", "advanced")
TRAIN_DEPENDENCY_CONSTRAINTS = {
    "accelerate": ">=1.1,<2",
    "bitsandbytes": ">=0.43,<1",
    "datasets": ">=2.21,<3",
    "peft": ">=0.12,<1",
    "safetensors": ">=0.4,<1",
    "torch": ">=2.4,<3",
    "transformers": ">=4.51,<5",
    "wandb": ">=0.18,<1",
}
