"""
Evaluation metrics for multi-label classification.

Computes accuracy, precision, recall, and F1 score by comparing
binarized predictions against ground truth labels.
"""

import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute accuracy, precision, recall, and F1 for multi-label output.

    Predictions are binarized using the threshold before comparison.
    All metrics are computed globally across all labels and samples,
    not per-label.

    Parameters
    ----------
    y_pred : np.ndarray of shape (n_samples, n_labels)
        Predicted probabilities from the sigmoid output layer.
    y_true : np.ndarray of shape (n_samples, n_labels)
        Ground truth binary labels (0 or 1).
    threshold : float
        Cutoff to binarize predictions. Values >= threshold become 1,
        values below become 0. Default is 0.5.

    Returns
    -------
    dict
        Dictionary with keys 'accuracy', 'precision', 'recall', 'f1',
        each mapped to a float rounded to 4 decimal places.

    Examples
    --------
    >>> y_pred = np.array([[0.8, 0.2], [0.6, 0.9]])
    >>> y_true = np.array([[1, 0], [1, 1]])
    >>> compute_metrics(y_pred, y_true)
    {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    preds = (y_pred >= threshold).astype(int)

    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    # Exact match accuracy — all labels must be correct for a sample to count
    correct = (preds == y_true).all(axis=1).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = correct / y_true.shape[0]

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
    }