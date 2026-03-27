"""
Dataset loading and partitioning utilities.

Downloads and loads the MNIST dataset for multi-class classification.
MNIST has 60,000 training images of handwritten digits (0-9),
each image is 28x28 pixels = 784 features when flattened.

Partitioning supports proportional splits for heterogeneous workers.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def generate_dataset(n_samples: int = 60000) -> tuple:
    """Load and prepare the MNIST dataset for multi-label classification.

    Downloads MNIST automatically on first run (cached after that).
    Pixel values are normalized from [0, 255] to [0, 1].
    Labels are one-hot encoded into 10 binary columns (one per digit).

    Parameters
    ----------
    n_samples : int
        Number of samples to use from MNIST. Max is 70000.
        Default is 60000 (standard MNIST training set size).

    Returns
    -------
    X_train : np.ndarray of shape (n_train, 784)
        Training feature matrix — flattened and normalized pixel values.
    X_test : np.ndarray of shape (n_test, 784)
        Test feature matrix.
    y_train : np.ndarray of shape (n_train, 10)
        One-hot encoded digit labels for training set.
    y_test : np.ndarray of shape (n_test, 10)
        One-hot encoded digit labels for test set.
    """
    print("[Data] Loading MNIST dataset (this may take a moment on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data[:n_samples].astype(np.float32) / 255.0  # normalize to [0,1]
    y_raw = mnist.target[:n_samples].astype(int)

    # One-hot encode labels: digit 3 → [0,0,0,1,0,0,0,0,0,0]
    lb = LabelBinarizer()
    y = lb.fit_transform(y_raw).astype(np.float32)

    print(f"[Data] Loaded {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} classes")
    return train_test_split(X, y, test_size=0.1, random_state=42)


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_workers: int,
    weights: list = None
) -> list:
    """Split data into shards for distributed workers.

    Supports proportional splitting so faster workers receive
    larger data shards (heterogeneity-aware partitioning).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full training feature matrix to partition.
    y : np.ndarray of shape (n_samples, n_labels)
        Corresponding label matrix.
    n_workers : int
        Number of worker nodes to partition data across.
    weights : list of float or None
        Relative speed weights per worker. If None, equal split applied.

    Returns
    -------
    list of tuple
        A list of (X_shard, y_shard) pairs, one per worker.
    """
    n_samples = X.shape[0]

    if weights is None:
        weights = [1.0] * n_workers

    total_weight = sum(weights)
    proportions = [w / total_weight for w in weights]

    sizes = [int(p * n_samples) for p in proportions]
    sizes[-1] += n_samples - sum(sizes)  # remainder to last worker

    shards, start = [], 0
    for size in sizes:
        shards.append((X[start:start + size], y[start:start + size]))
        start += size

    return shards