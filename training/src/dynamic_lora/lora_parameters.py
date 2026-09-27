"""Map PEFT LoRA parameter names to stable transformer layer keys.
Layer selection and signature capture share this naming contract."""

from __future__ import annotations

import re

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def is_lora_parameter(parameter_name: str) -> bool:
    """Return whether a parameter belongs to a PEFT LoRA adapter."""
    return "lora_" in parameter_name


def lora_layer_key(parameter_name: str) -> str:
    """Return a stable layer and module key for a PEFT LoRA parameter."""
    layer_match = LAYER_PATTERN.search(parameter_name)
    module_name = _module_name(parameter_name)
    if layer_match is None:
        return module_name
    return f"layer_{layer_match.group(1)}.{module_name}"


def _module_name(parameter_name: str) -> str:
    before_lora = parameter_name.split(".lora_", maxsplit=1)[0]
    return before_lora.rsplit(".", maxsplit=1)[-1]
