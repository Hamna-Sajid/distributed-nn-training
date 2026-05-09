"""
Dense (fully connected) layer implementation.

Each layer stores its input during the forward pass so it can compute
gradients during the backward pass (backpropagation).
"""

import numpy as np


class DenseLayer:
    """A fully connected (dense) neural network layer.

    Parameters
    ----------
    input_dim : int
        Number of input features to this layer.
    output_dim : int
        Number of output neurons in this layer.

    Attributes
    ----------
    W : np.ndarray of shape (input_dim, output_dim)
        Weight matrix, initialized with small random values.
    b : np.ndarray of shape (1, output_dim)
        Bias vector, initialized to zeros.
    input : np.ndarray or None
        Cached input from the last forward pass, used in backward pass.
    dW : np.ndarray or None
        Gradient of the loss with respect to W, computed in backward pass.
    db : np.ndarray or None
        Gradient of the loss with respect to b, computed in backward pass.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.input = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear transformation: Z = X @ W + b.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, input_dim)
            Input data or activations from the previous layer.

        Returns
        -------
        np.ndarray of shape (batch_size, output_dim)
            The pre-activation output of this layer.
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients with respect to input, W, and b.

        Parameters
        ----------
        grad_output : np.ndarray of shape (batch_size, output_dim)
            Gradient of the loss with respect to this layer's output,
            flowing back from the next layer or the loss function.

        Returns
        -------
        np.ndarray of shape (batch_size, input_dim)
            Gradient of the loss with respect to this layer's input,
            to be passed to the previous layer.
        """
        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def get_gradients(self) -> dict:
        """Return computed gradients as a dictionary.

        Returns
        -------
        dict
            Keys 'dW' and 'db' mapping to their respective np.ndarrays.
        """
        return {"dW": self.dW, "db": self.db}

    def update_params(self, dW: np.ndarray, db: np.ndarray, lr: float):
        """Apply gradient descent update to W and b.

        Parameters
        ----------
        dW : np.ndarray of shape (input_dim, output_dim)
            Averaged gradient for the weight matrix.
        db : np.ndarray of shape (1, output_dim)
            Averaged gradient for the bias vector.
        lr : float
            Learning rate (step size) for the update.
        """
        self.W -= lr * dW
        self.b -= lr * db