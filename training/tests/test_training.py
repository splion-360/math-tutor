"""Test shared LoRA trainer orchestration without loading real model weights.
These tests verify tokenizer formatting, model setup order, metadata, and probe wiring."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dynamic_lora.config import TrainingConfig
from dynamic_lora.training import _format_record, train_shared_lora


def training_record(record_id: str, difficulty: str) -> dict[str, object]:
    return {
        "id": record_id,
        "difficulty": difficulty,
        "topic": "geometry",
        "prompt": "Explain triangle area.",
        "manim_code": "from manim import *\nclass Lesson(Scene):\n    pass\n",
        "split": "train",
        "source": {"name": "unit-fixture", "reference": record_id},
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class FakeTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 99
    pad_token: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def save_pretrained(self, _path: Path) -> None:
        return None


def test_format_record_uses_chat_template_and_ends_with_eos() -> None:
    tokenizer = FakeTokenizer()

    formatted = _format_record(training_record("train-foundational-001", "foundational"), tokenizer)

    assert formatted["text"].startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nDifficulty: foundational" in formatted["text"]
    assert "<|im_start|>assistant\nfrom manim import *" in formatted["text"]
    assert formatted["text"].rstrip().endswith("<|im_end|>")
    assert "<|system|>" not in formatted["text"]


def test_seed_is_set_before_model_and_adapter_initialization(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    events: list[str] = []
    model_kwargs: dict[str, object] = {}
    training_args_kwargs: dict[str, object] = {}
    tokenized_batches: list[dict[str, object]] = []
    evaluated: list[object] = []
    probe_calls: list[dict[str, object]] = []
    callback_builds: list[dict[str, object]] = []
    added_callbacks: list[object] = []
    tokenizer = FakeTokenizer()

    class FakeSignatureCallback:
        def finalize(self) -> dict[str, object]:
            return {
                "enabled": True,
                "projection_dim": 64,
                "snapshot_count": 1,
            }

    class FakeModel:
        def get_nb_trainable_parameters(self) -> tuple[int, int]:
            return 10, 100

        def save_pretrained(self, _path: Path) -> None:
            return None

    class FakeTrainerCallback:
        pass

    class FakeDataset:
        @classmethod
        def from_list(cls, _records: list[dict[str, str]]) -> FakeDataset:
            return cls()

        def map(
            self,
            callback: Callable[[dict[str, list[str]]], dict[str, object]],
            **_kwargs: object,
        ) -> FakeDataset:
            tokenized_batches.append(callback({"text": ["sample"]}))
            return self

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.train_dataset = kwargs["train_dataset"]
            return None

        def train(self) -> SimpleNamespace:
            return SimpleNamespace(metrics={"train_loss": 1.5})

        def add_callback(self, callback: object) -> None:
            added_callbacks.append(callback)

        def evaluate(self, *, eval_dataset: object) -> dict[str, float]:
            evaluated.append(eval_dataset)
            return {"eval_loss": 1.25}

    def load_model(*_args: object, **kwargs: object) -> FakeModel:
        events.append("model")
        model_kwargs.update(kwargs)
        return FakeModel()

    transformers = SimpleNamespace(
        __version__="4.51.0",
        set_seed=lambda _seed: events.append("seed"),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        TrainingArguments=lambda **kwargs: training_args_kwargs.update(kwargs) or object(),
        DataCollatorForLanguageModeling=lambda **_kwargs: object(),
        Trainer=FakeTrainer,
        TrainerCallback=FakeTrainerCallback,
    )
    peft = SimpleNamespace(
        __version__="0.12.0",
        LoraConfig=lambda **_kwargs: object(),
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=lambda model, _config: events.append("adapter") or model,
    )
    modules = {
        "datasets": SimpleNamespace(__version__="2.21.0", Dataset=FakeDataset),
        "peft": peft,
        "torch": SimpleNamespace(__version__="2.4.0", bfloat16="bfloat16"),
        "transformers": transformers,
    }
    monkeypatch.setattr(
        "dynamic_lora.training.import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(
        "dynamic_lora.training.measure_lora_layer_gradient_energy",
        lambda **kwargs: probe_calls.append(kwargs)
        or {
            "enabled": True,
            "sample_count": 1,
            "top_k": 2,
            "grouping": "transformer_layer_from_lora_parameter_name",
            "observed_lora_parameters": 4,
            "energy_by_layer": {"layer_7.q_proj": 25.0},
            "ranked_layers": [{"layer": "layer_7.q_proj", "gradient_energy": 25.0}],
            "selected_layers": ["layer_7.q_proj"],
        },
    )
    monkeypatch.setattr(
        "dynamic_lora.training.gradient_probe_metrics",
        lambda _probe: {"gradient_probe/enabled": 1},
    )
    monkeypatch.setattr(
        "dynamic_lora.training.build_gradient_signature_callback",
        lambda **kwargs: callback_builds.append(kwargs) or FakeSignatureCallback(),
    )

    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_jsonl(
        train_path,
        [
            training_record("train-foundational-001", "foundational"),
            training_record("train-intermediate-001", "intermediate"),
            training_record("train-advanced-001", "advanced"),
        ],
    )
    write_jsonl(holdout_path, [{"id": "eval-001"}])
    config = TrainingConfig(
        train_path=train_path,
        holdout_path=holdout_path,
        output_dir=tmp_path / "adapter",
        metadata_path=tmp_path / "run.json",
        load_in_4bit=False,
        run_smoke_eval=True,
        layer_energy_probe_top_k=2,
        gradient_signature_dim=64,
    )

    plan = train_shared_lora(config)

    assert events.index("seed") < events.index("model") < events.index("adapter")
    assert model_kwargs["torch_dtype"] == "bfloat16"
    assert training_args_kwargs["report_to"] == []
    assert tokenized_batches[0]["input_ids"] == [[1, 2, 3, 99]]
    assert tokenized_batches[0]["attention_mask"] == [[1, 1, 1, 1]]
    assert plan.metadata["weights_loaded"] is True
    assert plan.metadata["parameter_budget"] == {
        "trainable_parameters": 10,
        "total_parameters": 100,
        "trainable_percent": 10.0,
    }
    assert plan.metadata["runtime_versions"]["transformers"] == "4.51.0"
    assert plan.metadata["tracking"]["active"] is False
    assert plan.metadata["tracking"]["reason"] == "missing_wandb_api_key"
    assert plan.metadata["smoke_eval"] == {
        "dataset": "tokenized_training_fixture",
        "purpose": "modal_trainer_smoke",
    }
    assert probe_calls[0]["sample_count"] == 16
    assert probe_calls[0]["top_k"] == 2
    assert plan.metadata["layer_energy_probe"]["selected_layers"] == ["layer_7.q_proj"]
    assert callback_builds[0]["selected_layers"] == ("layer_7.q_proj",)
    assert callback_builds[0]["projection_dim"] == 64
    assert callback_builds[0]["artifact_dir"] == config.output_dir / "gradient_signatures"
    assert callback_builds[0]["callback_base"] is FakeTrainerCallback
    assert len(added_callbacks) == 1
    assert plan.metadata["gradient_signatures"] == {
        "enabled": True,
        "projection_dim": 64,
        "snapshot_count": 1,
    }
    assert len(evaluated) == 1
    assert set(plan.metadata["runtime_versions"]) == {
        "accelerate",
        "bitsandbytes",
        "datasets",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    }


def test_tracking_metadata_is_written_before_heavy_training_imports(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    metadata_path = tmp_path / "run.json"
    write_jsonl(
        train_path,
        [
            training_record("train-foundational-001", "foundational"),
            training_record("train-intermediate-001", "intermediate"),
            training_record("train-advanced-001", "advanced"),
        ],
    )
    write_jsonl(holdout_path, [{"id": "eval-001"}])
    config = TrainingConfig(
        train_path=train_path,
        holdout_path=holdout_path,
        output_dir=tmp_path / "adapter",
        metadata_path=metadata_path,
        load_in_4bit=False,
    )

    def fail_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("dynamic_lora.training.import_module", fail_import)

    with pytest.raises(ModuleNotFoundError):
        train_shared_lora(config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["tracking"]["active"] is False
    assert metadata["tracking"]["reason"] == "missing_wandb_api_key"
    assert metadata["weights_loaded"] is False
