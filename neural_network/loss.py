"""
Loss functions for neural network training.

Binary cross-entropy is used for multi-label classification tasks where
each label is treated as an independent binary prediction.
"""

import numpy as np


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the mean binary cross-entropy loss over the batch.

    For each sample and each label independently:
        loss = -[y * log(p) + (1 - y) * log(1 - p)]

    Parameters
    ----------
    y_pred : np.ndarray of shape (batch_size, n_labels)
        Predicted probabilities in range (0, 1) from the sigmoid output layer.
    y_true : np.ndarray of shape (batch_size, n_labels)
        Ground truth binary labels (0 or 1).

    Returns
    -------
    float
        Scalar mean loss value averaged over all samples and labels.
    """
    eps = 1e-8  # prevent log(0)
    loss = -(
        y_true * np.log(y_pred + eps) +
        (1.0 - y_true) * np.log(1.0 - y_pred + eps)
    )
    return float(np.mean(loss))


def binary_cross_entropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute the gradient of BCE loss with respect to y_pred.

    Derivative:
        d_loss/d_pred = -(y / p) + (1 - y) / (1 - p), divided by N

    Parameters
    ----------
    y_pred : np.ndarray of shape (batch_size, n_labels)
        Predicted probabilities from the sigmoid output layer.
    y_true : np.ndarray of shape (batch_size, n_labels)
        Ground truth binary labels (0 or 1).

    Returns
    -------
    np.ndarray of shape (batch_size, n_labels)
        Gradient of the loss with respect to y_pred.
    """
    eps = 1e-8
    N = y_true.shape[0]
    return (
        -(y_true / (y_pred + eps)) +
        (1.0 - y_true) / (1.0 - y_pred + eps)
    ) / N