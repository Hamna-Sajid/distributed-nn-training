"""
Activation functions with forward and backward pass support.

Each activation caches values from the forward pass needed to compute
gradients during backpropagation.
"""

import numpy as np


class ReLU:
    """Rectified Linear Unit activation: f(x) = max(0, x).

    Attributes
    ----------
    mask : np.ndarray of bool or None
        Boolean mask of which inputs were positive during the forward pass.
        Used to gate gradients in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply ReLU element-wise: output is 0 where input <= 0.

        Parameters
        ----------
        X : np.ndarray
            Pre-activation values from a dense layer.

        Returns
        -------
        np.ndarray of same shape as X
            Activated values with negatives zeroed out.
        """
        self.mask = X > 0
        return X * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Pass gradient through only where input was positive.

        Parameters
        ----------
        grad_output : np.ndarray of same shape as forward input
            Upstream gradient flowing back from the next layer.

        Returns
        -------
        np.ndarray of same shape as grad_output
            Gradient zeroed out wherever the forward input was <= 0.
        """
        return grad_output * self.mask


class Sigmoid:
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x)).

    Used on the output layer for multi-label classification, where each
    output is an independent binary probability.

    Attributes
    ----------
    out : np.ndarray or None
        Sigmoid output cached from the forward pass, used in backward.
    """

    def __init__(self):
        self.out = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply sigmoid element-wise, clipping input for numerical stability.

        Parameters
        ----------
        X : np.ndarray
            Pre-activation values from the final dense layer.

        Returns
        -------
        np.ndarray of same shape as X
            Values in the range (0, 1) representing class probabilities.
        """
        X_clipped = np.clip(X, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-X_clipped))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient through sigmoid using cached output.

        The derivative of sigmoid is: sigma(x) * (1 - sigma(x)).

        Parameters
        ----------
        grad_output : np.ndarray of same shape as forward input
            Upstream gradient from the loss function.

        Returns
        -------
        np.ndarray of same shape as grad_output
            Gradient scaled by the sigmoid derivative.
        """
        return grad_output * self.out * (1.0 - self.out)