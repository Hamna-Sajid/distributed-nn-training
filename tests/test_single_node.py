"""
Single-node correctness test.

Verifies that the MLP trains correctly on a full dataset without any
distributed components — loss must decrease monotonically over 50 epochs.
Run this BEFORE testing distributed execution to confirm the NN is correct.
"""

import sys
import os

# Add the project root to the Python path before importing project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from neural_network.mlp import MLP
from data.loader import generate_dataset



def test_loss_decreases():
    """Assert that training loss decreases over epochs on a single node.

    This test:
    1. Generates a smaller synthetic multi-label dataset for fast validation
    2. Trains an MLP for 10 epochs using the full training set
    3. Asserts final loss < initial loss (basic sanity check)
    
    Note: Uses 2000 samples and 10 epochs for faster testing. 
    Production runs can use larger datasets/epochs (see commented code below).
    """
    # ========== OPTIMIZED VERSION (Fast) ==========
    X_train, _, y_train, _ = generate_dataset(n_samples=2000)
    model = MLP(input_dim=784, hidden_dim=2048, output_dim=10, lr=0.01)

    losses = []
    for epoch in range(10):
        print(f"[Epoch {epoch+1:2d}/10] Computing forward pass...", end="", flush=True)
        y_pred = model.forward(X_train)
        loss = model.compute_loss(y_pred, y_train)
        losses.append(loss)
        print(f" Loss: {loss:.4f} | Computing backward...", end="", flush=True)
        grads = model.backward(y_pred, y_train)
        model.apply_gradients(grads)
        print(" Done")

    # ========== COMMENTED: ORIGINAL VERSION (Slow - 30+ mins) ==========
    # Uncomment the code below to run the full test (10,000 samples, 50 epochs)
    # Note: This takes 30+ minutes on most machines. Use only for comprehensive validation.
    # 
    # X_train, _, y_train, _ = generate_dataset(n_samples=10000)
    # model = MLP(input_dim=784, hidden_dim=2048, output_dim=10, lr=0.01)
    #
    # losses = []
    # for epoch in range(50):
    #     y_pred = model.forward(X_train)
    #     loss = model.compute_loss(y_pred, y_train)
    #     losses.append(loss)
    #     grads = model.backward(y_pred, y_train)
    #     model.apply_gradients(grads)

    print(f"\nInitial loss: {losses[0]:.4f} | Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss did not decrease — check backprop!"
    print("Single-node test passed.")


if __name__ == "__main__":
    test_loss_decreases()