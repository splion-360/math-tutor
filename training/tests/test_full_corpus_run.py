"""Test full-corpus training inputs before model weights are loaded.
The tests protect row coverage and code-only supervised labels."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dynamic_lora.full_corpus_plan import build_full_corpus_plan
from dynamic_lora.full_corpus_run import (
    build_tokenized_splits,
    choose_monitor_modules,
    load_full_corpus_config,
    run_full_corpus,
    training_arguments_kwargs,
)


class CharacterTokenizer:
    eos_token = "!"
    eos_token_id = 33

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = "".join(f"[{message['role']}]{message['content']}" for message in messages)
        return rendered + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in value]}


def test_splits_tokenize_all_rows_and_mask_prompt_loss() -> None:
    records = [
        {
            "id": f"{difficulty}-{index}",
            "difficulty": difficulty,
            "topic": "geometry",
            "prompt": "Show a triangle.",
            "manim_code": "print(1)",
        }
        for difficulty in ("foundational", "intermediate", "advanced")
        for index in range(10)
    ]
    plan = build_full_corpus_plan(records, seed=42, epochs=3)

    training, validation = build_tokenized_splits(
        records, CharacterTokenizer(), plan, max_seq_length=3072
    )

    assert len(training) == 27
    assert len(validation) == 3
    assert {item["id"] for item in training} == set(plan.training_ids)
    assert {item["id"] for item in validation} == set(plan.validation_ids)
    for item in training + validation:
        labels = item["tokenized"]["labels"]
        assert labels.count(-100) > 0
        assert labels[-9:] == [ord(character) for character in "print(1)!"]


def test_training_arguments_use_epochs_and_validate_each_epoch(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoints"
    arguments = training_arguments_kwargs(output_dir=output_dir, seed=42, epochs=3)

    assert arguments["num_train_epochs"] == 3
    assert arguments["per_device_train_batch_size"] == 1
    assert arguments["gradient_accumulation_steps"] == 1
    assert arguments["eval_strategy"] == "epoch"
    assert arguments["save_strategy"] == "epoch"
    assert arguments["load_best_model_at_end"] is True
    assert "max_steps" not in arguments


def test_config_loads_three_epoch_full_corpus_run(tmp_path: Path) -> None:
    config_path = tmp_path / "full.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "artifact_root": "artifacts/full_corpus",
                "seed": 42,
                "epochs": 3,
                "max_seq_length": 3072,
                "learning_rate": 0.0002,
                "projection_dim": 256,
                "expected_training_count": 895,
                "expected_validation_count": 100,
            }
        ),
        encoding="utf-8",
    )

    config = load_full_corpus_config(config_path)

    assert config.epochs == 3
    assert config.train_path == tmp_path / "train.jsonl"
    assert config.artifact_root == tmp_path / "artifacts/full_corpus"
    assert config.projection_dim == 256
    assert config.expected_training_count == 895
    assert config.expected_validation_count == 100


def test_monitoring_spans_depth_and_all_projection_categories() -> None:
    categories = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = [
        f"base_model.model.model.layers.{layer}.self_attn.{category}.lora_B.default.weight"
        for layer in range(8)
        for category in categories
    ]

    selected = choose_monitor_modules(names, layer_count=4)

    assert len(selected) == 28
    assert {name.split(".")[0] for name in selected} == {
        "layer_0",
        "layer_2",
        "layer_5",
        "layer_7",
    }
    for category in categories:
        assert sum(name.endswith(f".{category}") for name in selected) == 4


def test_full_run_trains_three_epochs_and_probes_all_training_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        {
            "id": f"{difficulty}-{index}",
            "difficulty": difficulty,
            "topic": "geometry",
            "prompt": "Show a triangle.",
            "manim_code": "print(1)",
            "split": "train",
            "source": {"name": "test", "reference": str(index)},
        }
        for difficulty in ("foundational", "intermediate", "advanced")
        for index in range(10)
    ]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    holdout_path = tmp_path / "holdout.jsonl"
    holdout_path.write_text('{"id":"eval-one"}\n', encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": str(train_path),
                "holdout_path": str(holdout_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "seed": 42,
                "epochs": 3,
                "max_seq_length": 3072,
                "learning_rate": 0.0002,
                "projection_dim": 256,
            }
        ),
        encoding="utf-8",
    )
    config = load_full_corpus_config(config_path)
    observed: dict[str, Any] = {}

    class FakeDataset:
        def __init__(self, rows: list[dict[str, list[int]]]) -> None:
            self.rows = rows

        @classmethod
        def from_list(cls, rows: list[dict[str, list[int]]]) -> FakeDataset:
            return cls(rows)

        def __len__(self) -> int:
            return len(self.rows)

    class FakeModel:
        config = SimpleNamespace(use_cache=True)
        device = "cuda:0"

        def named_parameters(self) -> list[tuple[str, object]]:
            return [
                (
                    f"base_model.model.model.layers.{layer}.self_attn.{category}.lora_B.default.weight",
                    SimpleNamespace(requires_grad=True),
                )
                for layer in range(8)
                for category in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
            ]

        def get_nb_trainable_parameters(self) -> tuple[int, int]:
            return 100, 1000

        def save_pretrained(self, _path: Path) -> None:
            return None

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            observed["training_rows"] = len(kwargs["train_dataset"])
            observed["validation_rows"] = len(kwargs["eval_dataset"])
            self.state = SimpleNamespace(best_model_checkpoint="/tmp/checkpoint-54")

        def train(self) -> SimpleNamespace:
            return SimpleNamespace(global_step=81, metrics={"train_loss": 1.0, "epoch": 3.0})

        def evaluate(self) -> dict[str, float]:
            return {"eval_loss": 0.7}

    class FakeProbe:
        def capture(self, *, step: int, model: object) -> None:
            observed["probe_step"] = step

        def finalize(self) -> dict[str, object]:
            return {"snapshot_count": len(observed["probe_ids"]) * 28}

    def make_probe(**kwargs: Any) -> FakeProbe:
        observed["probe_ids"] = [item.record_id for item in kwargs["examples"]]
        observed["monitor_modules"] = kwargs["selected_layers"]
        return FakeProbe()

    class FakeWandbRun:
        url = "https://wandb.ai/test/run"

        def finish(self) -> None:
            return None

    class FakeTokenizer(CharacterTokenizer):
        pad_token: str | None = None

        def save_pretrained(self, _path: Path) -> None:
            return None

    modules = {
        "datasets": SimpleNamespace(Dataset=FakeDataset),
        "peft": SimpleNamespace(
            LoraConfig=lambda **_kwargs: object(),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            prepare_model_for_kbit_training=lambda model: model,
            get_peft_model=lambda model, _config: model,
        ),
        "torch": SimpleNamespace(bfloat16="bf16"),
        "transformers": SimpleNamespace(
            set_seed=lambda _seed: None,
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
            ),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: FakeModel()
            ),
            BitsAndBytesConfig=lambda **_kwargs: object(),
            TrainingArguments=lambda **kwargs: (
                observed.update({"training_args": kwargs}) or object()
            ),
            DataCollatorForSeq2Seq=lambda **_kwargs: object(),
            Trainer=FakeTrainer,
            TrainerCallback=object,
        ),
        "wandb": SimpleNamespace(init=lambda **_kwargs: FakeWandbRun()),
    }
    monkeypatch.setenv("WANDB_API_KEY", "unit-test")
    monkeypatch.setattr("dynamic_lora.full_corpus_run.import_module", lambda name: modules[name])
    monkeypatch.setattr(
        "dynamic_lora.full_corpus_run.build_fixed_prompt_signature_callback", make_probe
    )

    result = run_full_corpus(config)

    assert observed["training_rows"] == 27
    assert observed["validation_rows"] == 3
    assert observed["training_args"]["num_train_epochs"] == 3
    assert len(observed["probe_ids"]) == 27
    assert len(observed["monitor_modules"]) == 28
    assert observed["probe_step"] == 54
    assert result["dataset"]["training_count"] == 27
    assert result["gradient_probe"]["snapshot_count"] == 756
    assert result["training"]["probe_checkpoint_step"] == 54
