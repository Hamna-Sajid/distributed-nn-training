"""
Multi-Layer Perceptron (MLP) for multi-label classification.

Assembles dense layers and activations into a full network with
forward pass, loss computation, backward pass, and gradient extraction.
This is the model that will be replicated on each distributed worker.
"""

import numpy as np
from neural_network.layers import DenseLayer
from neural_network.activations import ReLU, Sigmoid
from neural_network.loss import binary_cross_entropy, binary_cross_entropy_grad


class MLP:
    """A shallow MLP: Input → Dense → ReLU → Dense → Sigmoid.

    Architecture (default):
        Input(20) → Dense(20→64) → ReLU → Dense(64→5) → Sigmoid

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of neurons in the hidden layer.
    output_dim : int
        Number of output labels (one sigmoid unit per label).
    lr : float
        Learning rate for gradient descent parameter updates.

    Attributes
    ----------
    layer1 : DenseLayer
        First fully connected layer (input → hidden).
    relu : ReLU
        ReLU activation after layer1.
    layer2 : DenseLayer
        Second fully connected layer (hidden → output).
    sigmoid : Sigmoid
        Sigmoid activation producing final probabilities.
    lr : float
        Learning rate stored for use during parameter updates.
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64,
                 output_dim: int = 5, lr: float = 0.01):
        self.layer1 = DenseLayer(input_dim, hidden_dim)
        self.relu = ReLU()
        self.layer2 = DenseLayer(hidden_dim, output_dim)
        self.sigmoid = Sigmoid()
        self.lr = lr

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward pass through the network.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, input_dim)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (batch_size, output_dim)
            Predicted probabilities for each label.
        """
        z1 = self.layer1.forward(X)
        a1 = self.relu.forward(z1)
        z2 = self.layer2.forward(a1)
        y_pred = self.sigmoid.forward(z2)
        return y_pred

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray of shape (batch_size, output_dim)
            Predicted probabilities from forward pass.
        y_true : np.ndarray of shape (batch_size, output_dim)
            Ground truth binary labels.

        Returns
        -------
        float
            Scalar loss value.
        """
        return binary_cross_entropy(y_pred, y_true)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """Run backpropagation and return all gradients.

        Computes gradients for all parameters by chaining the chain rule
        backward through: sigmoid → layer2 → relu → layer1.

        Parameters
        ----------
        y_pred : np.ndarray of shape (batch_size, output_dim)
            Predicted probabilities from the most recent forward pass.
        y_true : np.ndarray of shape (batch_size, output_dim)
            Ground truth binary labels.

        Returns
        -------
        dict
            Nested dict with keys 'layer1' and 'layer2', each containing
            'dW' and 'db' np.ndarrays for that layer's gradients.
        """
        # Gradient from loss into sigmoid
        d_loss = binary_cross_entropy_grad(y_pred, y_true)
        # Through sigmoid activation
        d_sig = self.sigmoid.backward(d_loss)
        # Through layer2
        d_layer2_input = self.layer2.backward(d_sig)
        # Through ReLU
        d_relu = self.relu.backward(d_layer2_input)
        # Through layer1
        self.layer1.backward(d_relu)

        return {
            "layer1": self.layer1.get_gradients(),
            "layer2": self.layer2.get_gradients(),
        }

    def apply_gradients(self, gradients: dict):
        """Update model parameters using provided (averaged) gradients.

        Parameters
        ----------
        gradients : dict
            Same structure as returned by backward():
            {'layer1': {'dW': ..., 'db': ...}, 'layer2': {'dW': ..., 'db': ...}}
        """
        self.layer1.update_params(
            gradients["layer1"]["dW"],
            gradients["layer1"]["db"],
            self.lr
        )
        self.layer2.update_params(
            gradients["layer2"]["dW"],
            gradients["layer2"]["db"],
            self.lr
        )

    def get_weights(self) -> dict:
        """Extract all model weights for broadcasting to workers.

        Returns
        -------
        dict
            Nested dict with 'layer1' and 'layer2', each containing
            'W' and 'b' as lists (JSON-serializable).
        """
        return {
            "layer1": {"W": self.layer1.W.tolist(), "b": self.layer1.b.tolist()},
            "layer2": {"W": self.layer2.W.tolist(), "b": self.layer2.b.tolist()},
        }

    def set_weights(self, weights: dict):
        """Load weights received from master into this model.

        Parameters
        ----------
        weights : dict
            Same structure as returned by get_weights().
        """
        self.layer1.W = np.array(weights["layer1"]["W"])
        self.layer1.b = np.array(weights["layer1"]["b"])
        self.layer2.W = np.array(weights["layer2"]["W"])
        self.layer2.b = np.array(weights["layer2"]["b"])