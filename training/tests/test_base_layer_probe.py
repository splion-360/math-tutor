"""Test no-update base-weight gradient measurement and ranking.
The tests distinguish base transformer weights from attached LoRA parameters."""

from __future__ import annotations

from dynamic_lora.base_layer_probe import (
    bootstrap_topk_frequency,
    compare_probe_rankings,
    measure_base_layer_gradient_energy,
)


class FakeGradient:
    def __init__(self, energy: float) -> None:
        self.energy = energy

    def detach(self) -> FakeGradient:
        return self

    def float(self) -> FakeGradient:
        return self

    def pow(self, exponent: int) -> FakeGradient:
        assert exponent == 2
        return self

    def sum(self) -> FakeGradient:
        return self

    def item(self) -> float:
        return self.energy


class FakeParameter:
    def __init__(self, elements: int) -> None:
        self.elements = elements
        self.grad: FakeGradient | None = None

    def numel(self) -> int:
        return self.elements


class FakeLoss:
    def backward(self) -> None:
        return None


class FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.zero_grad_calls = 0
        self.weights = {
            "model.layers.0.self_attn.q_proj.weight": FakeParameter(4),
            "model.layers.0.self_attn.v_proj.weight": FakeParameter(2),
            "model.layers.1.mlp.down_proj.weight": FakeParameter(8),
            "model.layers.1.mlp.down_proj.lora_B.default.weight": FakeParameter(2),
            "model.embed_tokens.weight": FakeParameter(10),
        }

    def __call__(self, **batch: object) -> dict[str, FakeLoss]:
        sample = int(batch["sample"])
        energies = (
            {"q_proj": 9.0, "v_proj": 4.0, "down_proj": 1.0},
            {"q_proj": 1.0, "v_proj": 4.0, "down_proj": 9.0},
        )[sample]
        for name, parameter in self.weights.items():
            module = name.split(".")[-2]
            parameter.grad = FakeGradient(energies.get(module, 100.0))
        return {"loss": FakeLoss()}

    def named_parameters(self) -> object:
        return iter(self.weights.items())

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        assert set_to_none is True
        self.zero_grad_calls += 1
        for parameter in self.weights.values():
            parameter.grad = None

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool) -> None:
        self.training = mode

    def parameters(self) -> object:
        return iter(())


def test_base_probe_ranks_base_weights_without_updates() -> None:
    model = FakeModel()
    result = measure_base_layer_gradient_energy(
        model=model,
        tokenized_dataset=[{"sample": 0}, {"sample": 1}],
        data_collator=lambda examples: examples[0],
        sample_count=2,
        top_k=2,
        target_modules=("q_proj", "v_proj", "down_proj"),
    )

    assert result["gradient_source"] == "base_weights"
    assert result["sample_indices"] == [0, 1]
    assert result["selected_modules"] == ["layer_0.q_proj", "layer_1.down_proj"]
    assert result["selected_layers"] == ["layer_0", "layer_1"]
    assert result["module_energy"] == {
        "layer_0.q_proj": 5.0,
        "layer_0.v_proj": 4.0,
        "layer_1.down_proj": 5.0,
    }
    assert result["layer_energy"] == {"layer_0": 9.0, "layer_1": 5.0}
    assert len(result["sample_module_energy"]) == 2
    assert model.zero_grad_calls == 3
    assert model.training is True


def test_bootstrap_frequency_exposes_unstable_rankings() -> None:
    samples = [
        {"layer_0.q_proj": 9.0, "layer_1.v_proj": 1.0},
        {"layer_0.q_proj": 1.0, "layer_1.v_proj": 9.0},
    ]

    frequency = bootstrap_topk_frequency(samples, top_k=1, repeats=100, seed=42)

    assert 0 < frequency["layer_0.q_proj"] < 1
    assert 0 < frequency["layer_1.v_proj"] < 1
    assert sum(frequency.values()) == 1.0
    assert frequency == bootstrap_topk_frequency(samples, top_k=1, repeats=100, seed=42)


def test_comparison_preserves_both_probe_rankings() -> None:
    comparison = compare_probe_rankings(
        base_modules=["layer_1.v_proj", "layer_0.q_proj"],
        lora_modules=["layer_0.q_proj", "layer_3.down_proj"],
    )

    assert comparison == {
        "base_top_modules": ["layer_1.v_proj", "layer_0.q_proj"],
        "lora_top_modules": ["layer_0.q_proj", "layer_3.down_proj"],
        "overlap": ["layer_0.q_proj"],
        "base_only": ["layer_1.v_proj"],
        "lora_only": ["layer_3.down_proj"],
    }
