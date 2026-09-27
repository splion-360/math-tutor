from __future__ import annotations

from shared_lora_baseline.gradient_probe import (
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
        "ranked_layers": [
            {"layer": "layer_7.q_proj", "gradient_energy": 25.0},
            {"layer": "layer_3.down_proj", "gradient_energy": 4.0},
        ],
        "selected_layers": ["layer_7.q_proj", "layer_3.down_proj"],
    }
    assert model.zero_grad_calls == 2


def test_gradient_probe_metrics_are_wandb_friendly() -> None:
    metrics = gradient_probe_metrics(
        {
            "enabled": True,
            "sample_count": 1,
            "top_k": 1,
            "observed_lora_parameters": 2,
            "energy_by_layer": {"layer_7.q_proj": 25.0},
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
    }
