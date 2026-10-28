# From an expression to an update

This walkthrough connects the pieces already covered in the video to the behavior of this codebase. It is also a small set of experiments to rerun when changing the engine.

## 1. Build an expression

```python
from micrograd.engine import Value

x = Value(3.0)
square = x * x
linear = 2 * x
y = square + linear

assert y.data == 15.0
assert x.grad == 0
```

Each operation produces a new `Value`. Its `data` holds the forward result, `_prev` references the operands, and `_backward` stores the local chain-rule calculation. The original `x` is shared by both branches and appears twice in the multiplication for `square`.

## 2. Inspect the graph

```text
x ──┬── x * x ──┐
    │            ├── add ── y
2 ──┴── 2 * x ──┘
```

For a rendered graph, run the helper-definition cell in `trace_graph.ipynb`, then call `draw_dot(y)`. The labels show each node's value and gradient. Call `draw_dot(y)` again after backpropagation to see the populated gradients. Graphviz's system executable must be installed to render the diagram.

## 3. Differentiate

```python
y.backward()
assert y.grad == 1
assert x.grad == 8.0
```

`backward()` orders the graph so every operation receives contributions from downstream paths before propagating to its operands. The derivative is `dy/dx = 2*x + 2`, which is 8 at `x = 3`.

The multiplication must add both contributions when its operands are the same object. Branches must also add their contributions rather than overwrite a shared node's gradient. The tests exercise both cases.

## 4. Take an SGD step

```python
learning_rate = 0.1
x.data -= learning_rate * x.grad

# Build a new forward graph using the updated value.
x.grad = 0
new_y = x * x + 2 * x
assert abs(new_y.data - 9.24) < 1e-10
```

The gradient points toward increasing `y`; subtracting a small multiple lowers it in this example. A network uses the same update for every weight and bias returned by `model.parameters()`.

## 5. Reset gradients and rebuild the graph

Gradient contributions use `+=`. Fresh graphs that share a leaf therefore accumulate into that leaf:

```python
x = Value(3.0)
(x * x).backward()
assert x.grad == 6.0
(x * x).backward()  # Fresh graph, same leaf.
assert x.grad == 12.0
```

That is useful when intentionally accumulating contributions. Ordinary training instead clears parameter gradients before each backward pass:

```python
from micrograd.nn import MLP

model = MLP(2, [1, 1])
for step in range(10):
    prediction = model([0.5, -1.0])  # Fresh graph on every iteration.
    loss = (prediction - 1.0)**2
    model.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        parameter.data -= 0.01 * parameter.grad
```

This example also exercises a one-neuron hidden layer, which now passes a list to the next layer internally. Final single-output predictions still return a `Value`.

Calling `backward()` repeatedly on the exact same graph has a separate limitation:

```python
x = Value(3.0)
y = (x * x) * 2
y.backward()
assert x.grad == 12.0
y.backward()
assert x.grad == 36.0  # Intermediate gradients were reused too.
```

`model.zero_grad()` only clears parameters, not intermediate nodes in an old graph. Rebuilding the graph each iteration avoids reusing those intermediate gradients. This extension preserves the original engine's small backward implementation and records this behavior in a regression test.

## 6. Try controlled experiments

- Pass one or three inputs to `MLP(2, [1, 1])` and inspect the dimension error.
- Compare `MLP(2, [1, 1])` with `MLP(2, [3, 1, 2])`; the final output is a scalar in the first case and a list in the second.
- Run `python train.py --seed 1337 --steps 100` twice, then try a different seed.
- Inspect ReLU at negative, zero, and positive inputs. This engine chooses a gradient of zero at zero, matching PyTorch.

Run `python -m pytest -q` after edits to check the analytic cases and PyTorch comparisons.
