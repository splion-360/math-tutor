"""Test subset-gradient ranking for unadapted base and matched LoRA weights.
The tests cover cancellation, whole-layer accounting, and parameter filtering."""

from __future__ import annotations

from dynamic_lora.base_layer_probe import (
    compare_probe_rankings,
    measure_base_layer_gradient_energy,
    measure_lora_subset_gradient_energy,
    selection_frequency,
)


class FakeGradient:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> FakeGradient:
        return self

    def float(self) -> FakeGradient:
        return self

    def pow(self, exponent: int) -> FakeGradient:
        assert exponent == 2
        return FakeGradient(self.value**2)

    def sum(self) -> FakeGradient:
        return self

    def item(self) -> float:
        return self.value


class FakeParameter:
    def __init__(self, elements: int) -> None:
        self.elements = elements
        self.grad: FakeGradient | None = None

    def numel(self) -> int:
        return self.elements


class FakeLoss:
    def __init__(self, model: FakeModel, sample: int) -> None:
        self.model = model
        self.sample = sample

    def backward(self) -> None:
        for name, value in self.model.sample_gradients[self.sample].items():
            parameter = self.model.weights[name]
            previous = parameter.grad.value if parameter.grad is not None else 0.0
            parameter.grad = FakeGradient(previous + value)


class FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.zero_grad_calls = 0
        self.weights = {
            "model.layers.0.self_attn.q_proj.weight": FakeParameter(4),
            "model.layers.0.self_attn.v_proj.weight": FakeParameter(2),
            "model.layers.0.input_layernorm.weight": FakeParameter(2),
            "model.layers.1.mlp.down_proj.weight": FakeParameter(8),
            "model.layers.0.self_attn.q_proj.lora_B.default.weight": FakeParameter(2),
            "model.layers.0.self_attn.v_proj.lora_B.default.weight": FakeParameter(2),
            "model.embed_tokens.weight": FakeParameter(10),
        }
        q, v, norm, down, lora_q, lora_v, embed = self.weights
        self.sample_gradients = (
            {q: 3.0, v: 1.0, norm: 2.0, down: 2.0, lora_q: 3.0, lora_v: 1.0, embed: 100.0},
            {q: -3.0, v: 1.0, norm: 2.0, down: 2.0, lora_q: -3.0, lora_v: 1.0, embed: 100.0},
        )

    def __call__(self, **batch: object) -> dict[str, FakeLoss]:
        return {"loss": FakeLoss(self, int(batch["sample"]))}

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


def test_base_probe_uses_subset_gradient_and_all_layer_weights() -> None:
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
    assert result["module_energy"] == {
        "layer_0.q_proj": 0.0,
        "layer_0.v_proj": 1.0,
        "layer_1.down_proj": 4.0,
    }
    assert result["layer_energy"] == {"layer_0": 5.0, "layer_1": 4.0}
    assert result["selected_modules"] == ["layer_1.down_proj", "layer_0.v_proj"]
    assert result["selected_layers"] == ["layer_0", "layer_1"]
    assert model.zero_grad_calls == 2
    assert model.training is True


def test_matched_lora_probe_uses_same_subset_score_but_only_lora_weights() -> None:
    model = FakeModel()
    result = measure_lora_subset_gradient_energy(
        model=model,
        tokenized_dataset=[{"sample": 0}, {"sample": 1}],
        data_collator=lambda examples: examples[0],
        sample_count=2,
        top_k=2,
        target_modules=("q_proj", "v_proj"),
    )

    assert result["gradient_source"] == "lora_parameters"
    assert result["module_energy"] == {"layer_0.q_proj": 0.0, "layer_0.v_proj": 1.0}
    assert result["layer_energy"] == {"layer_0": 1.0}


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


def test_selection_frequency_reports_disjoint_subset_agreement() -> None:
    frequency = selection_frequency(
        [["layer_0", "layer_1"], ["layer_0", "layer_2"], ["layer_0", "layer_2"]]
    )

    assert frequency == {"layer_0": 1.0, "layer_1": 1 / 3, "layer_2": 2 / 3}
