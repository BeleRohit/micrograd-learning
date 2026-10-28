import torch
import pytest
from micrograd.engine import Value


def test_reused_value():
    x = Value(3.0)
    y = x * x
    y.backward()
    assert y.data == 9.0
    assert x.grad == 6.0


def test_shared_intermediate_receives_all_branch_gradients():
    x = Value(2.0)
    shared = x * x
    y = shared * 3 + shared**2
    y.backward()
    assert y.data == 28.0
    assert shared.grad == 11.0
    assert x.grad == 44.0


@pytest.mark.parametrize("value", [-2.0, 0.0, 2.0])
def test_relu_matches_torch_at_and_around_zero(value):
    x = Value(value)
    y = x.relu()
    y.backward()
    reference = torch.tensor(value, dtype=torch.double, requires_grad=True)
    result = reference.relu()
    result.backward()
    assert y.data == result.item()
    assert x.grad == reference.grad.item()


def test_leaf_gradients_accumulate_across_fresh_graphs():
    x = Value(3.0)
    (x * x).backward()
    assert x.grad == 6.0
    (x * x).backward()
    assert x.grad == 12.0
    x.grad = 0
    (x * x).backward()
    assert x.grad == 6.0


def test_same_graph_reuses_intermediate_gradients():
    # Document the original engine's limitation rather than imply PyTorch parity.
    x = Value(3.0)
    y = (x * x) * 2
    y.backward()
    assert x.grad == 12.0
    y.backward()
    assert x.grad == 36.0

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
