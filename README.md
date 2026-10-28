# micrograd — learning extensions

A personal learning extension of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd): a tiny scalar autograd engine and neural network library in pure Python.

I watched the [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) end to end and learned the underlying concepts: computation graphs, the chain rule, backpropagation, neurons, and training with gradient descent. This repository is my follow-up work, applying that understanding through small correctness fixes, tests, and reproducible experiments.

The original engine and neural network implementation are Karpathy's work. The changes here build on that foundation as part of my learning process. This is a separate repository on my account; the upstream history and MIT attribution are preserved.

## Improvements made

| Area | Original behavior | Change in this repository |
| --- | --- | --- |
| Input dimensions | `zip` silently truncated mismatched inputs and weights. | Neurons raise a clear `ValueError` for too few or too many inputs. |
| Narrow hidden layers | A one-neuron hidden layer returned a scalar that the next layer could not consume. | Hidden layers pass lists internally; single-output public calls still return a `Value`. |
| Gradient coverage | Two expressions were compared against PyTorch. | Added tests for reused values, branching, ReLU at zero, gradient accumulation, and narrow networks. |
| Learning walkthrough | Examples lived mainly in notebooks. | Added a walkthrough connecting an expression, its graph, gradients, and an SGD update. |
| Reproducible training | The notebook seeded RNGs, but there was no standalone training command. | Added a seeded command-line demo, explicit dataset seeds, and installable dependency groups. |
| Automated checks | Tests ran manually. | Added GitHub Actions checks on Python 3.10 and 3.13 for pushes and pull requests. |

The goal is to keep the core small and readable while making its behavior easier to verify.

## Setup

Use Python 3.10 or newer for the development and notebook dependencies below. The scalar engine itself has no third-party runtime dependencies.

```bash
git clone https://github.com/BeleRohit/micrograd-learning.git
cd micrograd-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,demo]'
```

Install this checkout with `-e .` to use only the engine and neural network library. Installing `micrograd` directly from PyPI gives you the upstream package rather than these extensions.

## Quick example

```python
from micrograd.engine import Value

x = Value(3.0)
y = x * x + 2 * x
y.backward()

print(y.data)  # 15.0
print(x.grad)  # 8.0: dy/dx = 2*x + 2
```

For the calculation graph, an update step, and gradient accumulation behavior, see [the walkthrough](docs/walkthrough.md).

## Train without Jupyter

```bash
python train.py --seed 1337 --steps 100
```

This trains the original `MLP(2, [16, 16, 1])` architecture on 100 noisy moons samples using hinge loss, L2 regularization, and SGD. The script prints the initial loss, progress, and final loss and accuracy after the last update. These are training-set metrics; they do not measure generalization.

The seed fixes the dataset and weight initialization. Repeat the same command in the same dependency environment to reproduce the experiment; different dependency versions or platforms can introduce numerical differences.

### What I observed

For the default seed and 100 steps, the local run started at a loss of `0.895844` and 50% training accuracy. After the final update, loss was `0.010978` and training accuracy was 100%. This shows that the small network can fit this dataset; a separate held-out set would be needed to assess generalization.

The local test suite passed all 23 cases. The run used Python 3.13.7, PyTorch 2.9.0, NumPy 2.5.1, and scikit-learn 1.9.0. These versions record the experiment environment rather than lock dependencies for every platform.

## Notebooks and visualization

```bash
jupyter lab
```

- `demo.ipynb`: train the moons classifier and plot its decision boundary. Run from the first cell to reset the seeds.
- `trace_graph.ipynb`: draw computation graphs with values and gradients.

The graph notebook also needs the Graphviz system executable. On macOS, install it with `brew install graphviz`; on Debian/Ubuntu, use `sudo apt-get install graphviz`. The Python `graphviz` package is included in the `demo` dependency group.

## Tests

```bash
python -m pytest -q
```

Tests cover input errors, single-neuron hidden layers, public output shapes, parameter gradient clearing, shared operands, branching, and ReLU around zero. PyTorch provides reference outputs and gradients for the original expressions and the narrow-network regression tests.

GitHub Actions runs the suite automatically on Python 3.10 and 3.13. PyTorch is a test dependency, not part of the engine.

## Project layout

```text
micrograd/engine.py       Scalar Value objects and reverse-mode autodiff
micrograd/nn.py           Neuron, Layer, MLP, and parameter helpers
train.py                 Reproducible command-line training example
demo.ipynb               Training and decision-boundary visualization
trace_graph.ipynb        Graphviz computation-graph visualization
test/                    Engine and neural network regression tests
docs/walkthrough.md      Expression → graph → gradients → update
.github/workflows/       Automated test configuration
```

## Scope and limitations

Operations act on individual scalars. There are no tensors, GPU acceleration, or production training utilities. The recursive graph traversal can hit Python's recursion limit on very deep graphs.

Call `model.zero_grad()` before backpropagation for each new training step, and rebuild the forward graph each time. Leaf gradients accumulate across fresh graphs. Repeating `backward()` on the same graph also reuses intermediate gradients, so it does not behave like PyTorch's retained-graph backpropagation. The walkthrough demonstrates this distinction.

## Credits and license

- Original project: [karpathy/micrograd](https://github.com/karpathy/micrograd).
- Original author: Andrej Karpathy.
- Learning extensions maintained by [BeleRohit](https://github.com/BeleRohit).
- [MIT license](LICENSE), with the original copyright notice retained.
