"""Test deterministic LoRA gradient energy measurement behavior.
These tests cover layer ranking, metric flattening, and device-independent fake gradients."""

from __future__ import annotations

from dynamic_lora.layer_selection import (
    _probe_indices,
    gradient_probe_metrics,
    measure_lora_layer_gradient_energy,
)


class FakeGradient:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> FakeGradient:
        return self

    def float(self) -> FakeGradient:
        return self

    def pow(self, exponent: int) -> FakeGradient:
        return FakeGradient(self.value**exponent)

    def sum(self) -> FakeGradient:
        return self

    def item(self) -> float:
        return self.value


class FakeParameter:
    def __init__(self, gradient_energy: float) -> None:
        self.grad = FakeGradient(gradient_energy**0.5)

    def numel(self) -> int:
        return 2


class FakeLoss:
    def backward(self) -> None:
        return None


class FakeModel:
    def __init__(self) -> None:
        self.zero_grad_calls = 0

    def __call__(self, **_batch: object) -> dict[str, FakeLoss]:
        return {"loss": FakeLoss()}

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        assert set_to_none is True
        self.zero_grad_calls += 1

    def parameters(self) -> iter:
        return iter(())

    def named_parameters(self) -> iter:
        return iter(
            (
                (
                    "base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight",
                    FakeParameter(9.0),
                ),
                (
                    "base_model.model.model.layers.7.self_attn.q_proj.lora_B.default.weight",
                    FakeParameter(16.0),
                ),
                (
                    "base_model.model.model.layers.3.mlp.down_proj.lora_A.default.weight",
                    FakeParameter(4.0),
                ),
                ("base_model.model.embed_tokens.weight", FakeParameter(100.0)),
            )
        )


class FakeDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": [index + 1]}


def test_measure_lora_layer_gradient_energy_ranks_layers_and_records_details() -> None:
    model = FakeModel()

    probe = measure_lora_layer_gradient_energy(
        model=model,
        tokenized_dataset=FakeDataset(),
        data_collator=lambda examples: {
            "input_ids": [example["input_ids"] for example in examples]
        },
        sample_count=1,
        top_k=2,
    )

    assert probe == {
        "enabled": True,
        "sample_count": 1,
        "top_k": 2,
        "grouping": "transformer_layer_from_lora_parameter_name",
        "observed_lora_parameters": 3,
        "energy_by_layer": {
            "layer_3.down_proj": 4.0,
            "layer_7.q_proj": 25.0,
        },
        "energy_per_parameter_by_layer": {
            "layer_3.down_proj": 2.0,
            "layer_7.q_proj": 6.25,
        },
        "category_mean_energy": {"down_proj": 4.0, "q_proj": 25.0},
        "category_mean_energy_per_parameter": {"down_proj": 2.0, "q_proj": 6.25},
        "relative_energy_by_layer": {
            "layer_3.down_proj": 1.0,
            "layer_7.q_proj": 1.0,
        },
        "ranked_layers": [
            {"layer": "layer_7.q_proj", "gradient_energy": 25.0},
            {"layer": "layer_3.down_proj", "gradient_energy": 4.0},
        ],
        "selected_layers": ["layer_7.q_proj", "layer_3.down_proj"],
    }
    assert model.zero_grad_calls == 2


def test_probe_averages_per_prompt_energy_and_reports_category_relative_energy() -> None:
    class VariableModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.training = True
            self.seen_inputs: list[int] = []
            self.q_early = FakeParameter(0.0)
            self.q_late = FakeParameter(0.0)
            self.v_early = FakeParameter(0.0)

        def __call__(self, **batch: object) -> dict[str, FakeLoss]:
            assert self.training is False
            input_ids = batch["input_ids"]
            assert isinstance(input_ids, list)
            assert len(input_ids) == 1
            sample_id = input_ids[0][0]
            self.seen_inputs.append(sample_id)
            self.q_early.grad = FakeGradient(float(sample_id))
            self.q_late.grad = FakeGradient(1.0)
            self.v_early.grad = FakeGradient(2.0)
            return {"loss": FakeLoss()}

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool) -> None:
            self.training = mode

        def named_parameters(self) -> iter:
            return iter(
                (
                    ("model.layers.0.self_attn.q_proj.lora_A.weight", self.q_early),
                    ("model.layers.1.self_attn.q_proj.lora_A.weight", self.q_late),
                    ("model.layers.0.self_attn.v_proj.lora_A.weight", self.v_early),
                )
            )

    model = VariableModel()
    probe = measure_lora_layer_gradient_energy(
        model=model,
        tokenized_dataset=FakeDataset(),
        data_collator=lambda examples: {
            "input_ids": [example["input_ids"] for example in examples]
        },
        sample_count=2,
        top_k=2,
    )

    assert model.seen_inputs == [1, 2]
    assert probe["energy_by_layer"] == {
        "layer_0.q_proj": 2.5,
        "layer_0.v_proj": 4.0,
        "layer_1.q_proj": 1.0,
    }
    assert probe["selected_layers"] == ["layer_0.v_proj", "layer_0.q_proj"]
    assert probe["category_mean_energy"] == {"q_proj": 1.75, "v_proj": 4.0}
    assert probe["category_mean_energy_per_parameter"] == {
        "q_proj": 0.875,
        "v_proj": 2.0,
    }
    assert probe["relative_energy_by_layer"] == {
        "layer_0.q_proj": 2.5 / 1.75,
        "layer_0.v_proj": 1.0,
        "layer_1.q_proj": 1.0 / 1.75,
    }
    assert probe["energy_per_parameter_by_layer"] == {
        "layer_0.q_proj": 1.25,
        "layer_0.v_proj": 2.0,
        "layer_1.q_proj": 0.5,
    }
    assert model.zero_grad_calls == 3
    assert model.training is True


def test_probe_indices_spread_across_dataset() -> None:
    assert _probe_indices(dataset_size=100, sample_count=5) == [0, 24, 49, 74, 99]
    assert _probe_indices(dataset_size=3, sample_count=16) == [0, 1, 2]


def test_lora_gradient_energy_metrics_are_wandb_friendly() -> None:
    metrics = gradient_probe_metrics(
        {
            "enabled": True,
            "sample_count": 1,
            "top_k": 1,
            "observed_lora_parameters": 2,
            "energy_by_layer": {"layer_7.q_proj": 25.0},
            "category_mean_energy": {"q_proj": 25.0},
            "category_mean_energy_per_parameter": {"q_proj": 6.25},
            "selected_layers": ["layer_7.q_proj"],
        }
    )

    assert metrics == {
        "gradient_probe/enabled": 1,
        "gradient_probe/top_k": 1,
        "gradient_probe/sample_count": 1,
        "gradient_probe/observed_lora_parameters": 2,
        "gradient_probe/selected_layers": "layer_7.q_proj",
        "gradient_probe/energy/layer_7.q_proj": 25.0,
        "gradient_probe/category_mean_energy/q_proj": 25.0,
        "gradient_probe/category_mean_energy_per_parameter/q_proj": 6.25,
    }
