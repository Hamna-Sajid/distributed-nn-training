"""
Dataset generation and partitioning utilities.

Uses sklearn's synthetic multi-label generator so no external files
are needed. Partitioning supports proportional splits for heterogeneous
workers (faster workers get larger shards).
"""

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split


def generate_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 5,
    n_labels: int = 2,
    random_state: int = 42
) -> tuple:
    """Generate a synthetic multi-label classification dataset.

    Parameters
    ----------
    n_samples : int
        Total number of data samples to generate.
    n_features : int
        Number of input features per sample.
    n_classes : int
        Number of distinct output labels (each is binary).
    n_labels : int
        Average number of labels active per sample.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray of shape (n_train, n_features)
        Training feature matrix (80% of data).
    X_test : np.ndarray of shape (n_test, n_features)
        Test feature matrix (20% of data).
    y_train : np.ndarray of shape (n_train, n_classes)
        Binary label matrix for training set.
    y_test : np.ndarray of shape (n_test, n_classes)
        Binary label matrix for test set.
    """
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=random_state
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_workers: int,
    weights: list = None
) -> list:
    """Split data into shards for distributed workers.

    Supports proportional splitting so that faster workers receive
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
        Relative speed weights per worker (e.g., [1.0, 2.0] means
        worker 1 gets twice the data of worker 0). If None, equal
        split is applied.

    Returns
    -------
    list of tuple
        A list of (X_shard, y_shard) pairs, one per worker,
        where X_shard is np.ndarray of shape (shard_size, n_features).
    """
    n_samples = X.shape[0]

    if weights is None:
        weights = [1.0] * n_workers

    total_weight = sum(weights)
    proportions = [w / total_weight for w in weights]

    # Compute shard sizes, ensuring they sum to n_samples
    sizes = [int(p * n_samples) for p in proportions]
    sizes[-1] += n_samples - sum(sizes)  # remainder goes to last worker

    shards = []
    start = 0
    for size in sizes:
        end = start + size
        shards.append((X[start:end], y[start:end]))
        start = end

    return shards