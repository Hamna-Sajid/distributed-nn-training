"""
Multi-Layer Perceptron (MLP) for multi-class classification.

Architecture for MNIST:
    Input(784) → Dense(784→2048) → ReLU
               → Dense(2048→2048) → ReLU
               → Dense(2048→2048) → ReLU
               → Dense(2048→10) → Sigmoid
"""

import numpy as np
from neural_network.layers import DenseLayer
from neural_network.activations import ReLU, Sigmoid
from neural_network.loss import binary_cross_entropy, binary_cross_entropy_grad


class MLP:
    """A deep MLP with 3 hidden layers for MNIST digit classification.

    Parameters
    ----------
    input_dim : int
        Number of input features. 784 for MNIST (28x28 pixels).
    hidden_dim : int
        Number of neurons in each hidden layer. Default 2048.
    output_dim : int
        Number of output classes. 10 for MNIST (digits 0-9).
    lr : float
        Learning rate for gradient descent updates.

    Attributes
    ----------
    layer1 : DenseLayer
        Input → Hidden layer 1  (784 → 2048)
    layer2 : DenseLayer
        Hidden layer 1 → Hidden layer 2  (2048 → 2048)
    layer3 : DenseLayer
        Hidden layer 2 → Hidden layer 3  (2048 → 2048)
    layer4 : DenseLayer
        Hidden layer 3 → Output  (2048 → 10)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 2048,
                 output_dim: int = 10, lr: float = 0.01):
        self.layer1 = DenseLayer(input_dim, hidden_dim)
        self.relu1  = ReLU()
        self.layer2 = DenseLayer(hidden_dim, hidden_dim)
        self.relu2  = ReLU()
        self.layer3 = DenseLayer(hidden_dim, hidden_dim)
        self.relu3  = ReLU()
        self.layer4 = DenseLayer(hidden_dim, output_dim)
        self.sigmoid = Sigmoid()
        self.lr = lr

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward pass through all 4 layers.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, input_dim)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (batch_size, output_dim)
            Predicted probabilities for each class.
        """
        z1 = self.layer1.forward(X)
        a1 = self.relu1.forward(z1)
        z2 = self.layer2.forward(a1)
        a2 = self.relu2.forward(z2)
        z3 = self.layer3.forward(a2)
        a3 = self.relu3.forward(z3)
        z4 = self.layer4.forward(a3)
        return self.sigmoid.forward(z4)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray of shape (batch_size, output_dim)
            Predicted probabilities.
        y_true : np.ndarray of shape (batch_size, output_dim)
            Ground truth one-hot labels.

        Returns
        -------
        float
            Scalar loss value.
        """
        return binary_cross_entropy(y_pred, y_true)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """Run backpropagation through all 4 layers.

        Parameters
        ----------
        y_pred : np.ndarray of shape (batch_size, output_dim)
            Predictions from the most recent forward pass.
        y_true : np.ndarray of shape (batch_size, output_dim)
            Ground truth one-hot labels.

        Returns
        -------
        dict
            Keys 'layer1' through 'layer4', each containing 'dW' and 'db'.
        """
        d = binary_cross_entropy_grad(y_pred, y_true)
        d = self.sigmoid.backward(d)
        d = self.layer4.backward(d)
        d = self.relu3.backward(d)
        d = self.layer3.backward(d)
        d = self.relu2.backward(d)
        d = self.layer2.backward(d)
        d = self.relu1.backward(d)
        self.layer1.backward(d)

        return {
            "layer1": self.layer1.get_gradients(),
            "layer2": self.layer2.get_gradients(),
            "layer3": self.layer3.get_gradients(),
            "layer4": self.layer4.get_gradients(),
        }

    def apply_gradients(self, gradients: dict):
        """Apply averaged gradients to all layers.

        Parameters
        ----------
        gradients : dict
            Same structure as returned by backward().
        """
        for i, layer in enumerate([self.layer1, self.layer2,
                                    self.layer3, self.layer4], 1):
            layer.update_params(
                gradients[f"layer{i}"]["dW"],
                gradients[f"layer{i}"]["db"],
                self.lr
            )

    def get_weights(self) -> dict:
        """Extract all model weights for broadcasting.

        Returns
        -------
        dict
            Nested dict with 'layer1' through 'layer4',
            each containing 'W' and 'b' as lists.
        """
        return {
            f"layer{i}": {"W": layer.W.tolist(), "b": layer.b.tolist()}
            for i, layer in enumerate([self.layer1, self.layer2,
                                        self.layer3, self.layer4], 1)
        }

    def set_weights(self, weights: dict):
        """Load weights received from master.

        Parameters
        ----------
        weights : dict
            Same structure as returned by get_weights().
        """
        for i, layer in enumerate([self.layer1, self.layer2,
                                    self.layer3, self.layer4], 1):
            layer.W = np.array(weights[f"layer{i}"]["W"])
            layer.b = np.array(weights[f"layer{i}"]["b"])