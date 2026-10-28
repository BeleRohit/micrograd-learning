"""Train the original moons classifier without Jupyter: python train.py."""

import argparse
import random

from sklearn.datasets import make_moons

from micrograd.nn import MLP


def loss(model, inputs, labels):
    scores = [model(row) for row in inputs]
    margin = sum((1 - label * score).relu()
                 for label, score in zip(labels, scores)) / len(labels)
    regularization = 1e-4 * sum(p * p for p in model.parameters())
    accuracy = sum((label > 0) == (score.data > 0)
                   for label, score in zip(labels, scores)) / len(labels)
    return margin + regularization, accuracy


def train(seed=1337, steps=100):
    if steps < 1:
        raise ValueError("steps must be at least 1")
    random.seed(seed)
    inputs, labels = make_moons(n_samples=100, noise=0.1, random_state=seed)
    labels = labels * 2 - 1
    model = MLP(2, [16, 16, 1])
    initial_loss, initial_accuracy = loss(model, inputs, labels)
    print(f"seed={seed} steps={steps} parameters={len(model.parameters())}")
    print(f"initial loss={initial_loss.data:.6f} accuracy={initial_accuracy:.1%}")

    for step in range(steps):
        total_loss, accuracy = loss(model, inputs, labels)
        model.zero_grad()
        total_loss.backward()
        learning_rate = 1.0 - 0.9 * step / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={total_loss.data:.6f} accuracy={accuracy:.1%}")

    final_loss, final_accuracy = loss(model, inputs, labels)
    print(f"final loss={final_loss.data:.6f} accuracy={final_accuracy:.1%}")
    return model, final_loss.data, final_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    train(seed=args.seed, steps=args.steps)


if __name__ == "__main__":
    main()
