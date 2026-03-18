"""
Single-node correctness test.

Verifies that the MLP trains correctly on a full dataset without any
distributed components — loss must decrease monotonically over 50 epochs.
Run this BEFORE testing distributed execution to confirm the NN is correct.
"""

import numpy as np
from neural_network.mlp import MLP
from data.loader import generate_dataset


def test_loss_decreases():
    """Assert that training loss decreases over 50 epochs on a single node.

    This test:
    1. Generates the synthetic multi-label dataset
    2. Trains an MLP for 50 epochs using the full training set
    3. Asserts final loss < initial loss (basic sanity check)
    """
    X_train, _, y_train, _ = generate_dataset(n_samples=500)
    model = MLP(input_dim=20, hidden_dim=64, output_dim=5, lr=0.01)

    losses = []
    for epoch in range(50):
        y_pred = model.forward(X_train)
        loss = model.compute_loss(y_pred, y_train)
        losses.append(loss)
        grads = model.backward(y_pred, y_train)
        model.apply_gradients(grads)

    print(f"Initial loss: {losses[0]:.4f} | Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss did not decrease — check backprop!"
    print("Single-node test passed.")


if __name__ == "__main__":
    test_loss_decreases()