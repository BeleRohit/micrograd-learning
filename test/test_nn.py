import random

import pytest
import torch

from micrograd.engine import Value
from micrograd.nn import Layer, MLP, Neuron


@pytest.mark.parametrize("size", [0, 1, 3])
@pytest.mark.parametrize("factory", [lambda: Neuron(2), lambda: Layer(2, 3),
                                     lambda: MLP(2, [3, 1])])
def test_input_size_mismatch(factory, size):
    with pytest.raises(ValueError, match=f"expected 2 inputs, got {size}"):
        factory()([1.0] * size)


def test_single_output_layer_keeps_public_scalar_api():
    layer = Layer(2, 1)
    assert isinstance(layer([1.0, 2.0]), Value)
    assert isinstance(layer([1.0, 2.0], as_list=True), list)


@pytest.mark.parametrize("widths", [[1, 1], [1, 3, 1], [3, 1, 2]])
def test_narrow_hidden_layers_match_torch(widths):
    random.seed(1337)
    model = MLP(2, widths)
    # Positive weights keep ReLU active so every layer's gradient is exercised.
    for i, p in enumerate(model.parameters()):
        p.data = 0.05 * (i % 5 + 1)

    actual = model([0.5, 1.5])
    outputs = [actual] if isinstance(actual, Value) else actual
    sum(outputs).backward()

    reference_params = []
    expected = torch.tensor([0.5, 1.5], dtype=torch.double)
    for layer in model.layers:
        values = []
        for neuron in layer.neurons:
            weights = [torch.tensor(p.data, dtype=torch.double, requires_grad=True)
                       for p in neuron.w]
            bias = torch.tensor(neuron.b.data, dtype=torch.double, requires_grad=True)
            reference_params.extend(weights + [bias])
            value = sum((w * x for w, x in zip(weights, expected)), bias)
            values.append(value.relu() if neuron.nonlin else value)
        expected = torch.stack(values)
    expected.sum().backward()

    assert [v.data for v in outputs] == pytest.approx(expected.detach().tolist())
    assert [p.grad for p in model.parameters()] == pytest.approx(
        [p.grad.item() for p in reference_params])
    assert isinstance(actual, Value) if widths[-1] == 1 else isinstance(actual, list)


def test_zero_grad_clears_parameters_before_next_forward_pass():
    model = MLP(2, [1, 1])
    for p in model.parameters():
        p.data = 0.5
    model([1.0, 2.0]).backward()
    first = [p.grad for p in model.parameters()]
    assert all(g != 0 for g in first)
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())
    model([1.0, 2.0]).backward()
    assert [p.grad for p in model.parameters()] == pytest.approx(first)
