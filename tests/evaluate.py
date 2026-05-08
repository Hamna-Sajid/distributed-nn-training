"""
Evaluation metrics for multi-label classification.

Computes accuracy, precision, recall, and F1 score by comparing
binarized predictions against ground truth labels.
"""

import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute accuracy, precision, recall, and F1 for multi-class classification.

    For one-hot encoded labels (multi-class), compares argmax predictions
    to argmax true labels. Computes weighted average precision/recall/F1
    across all classes using macro averaging.

    Parameters
    ----------
    y_pred : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities from the output layer.
    y_true : np.ndarray of shape (n_samples, n_classes)
        Ground truth one-hot encoded labels.
    threshold : float
        Unused for multi-class (kept for API compatibility).

    Returns
    -------
    dict
        Dictionary with keys 'accuracy', 'precision', 'recall', 'f1',
        each mapped to a float rounded to 4 decimal places.

    Examples
    --------
    >>> y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0]])
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])
    >>> compute_metrics(y_pred, y_true)
    {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    # For multi-class classification, use argmax to get predicted and true class indices
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    
    # Accuracy: percentage of correct predictions
    accuracy = np.mean(pred_classes == true_classes)
    
    # Compute precision, recall, F1 using macro averaging (per-class then average)
    n_classes = y_true.shape[1]
    precisions, recalls, f1s = [], [], []
    
    for class_idx in range(n_classes):
        # For each class, treat as binary classification
        pred_binary = (pred_classes == class_idx).astype(int)
        true_binary = (true_classes == class_idx).astype(int)
        
        tp = np.sum((pred_binary == 1) & (true_binary == 1))
        fp = np.sum((pred_binary == 1) & (true_binary == 0))
        fn = np.sum((pred_binary == 0) & (true_binary == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Macro average across classes
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(avg_precision), 4),
        "recall":    round(float(avg_recall), 4),
        "f1":        round(float(avg_f1), 4),
    }