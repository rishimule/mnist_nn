"""
network.py

This module defines the NeuralNetwork class which manages the network's layers,
forward and backward propagation, and training process.
"""

from typing import List, Callable
import numpy as np

class NeuralNetwork:
    """
    NeuralNetwork class to build and train a modular neural network.

    Attributes:
        layers (List): List of layers in the network.
        loss (Callable): Loss function to evaluate performance.
        loss_derivative (Callable): Derivative of the loss function.
    """
    def __init__(self, layers: List,
                 loss: Callable[[np.ndarray, np.ndarray], float],
                 loss_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """
        Initialize the Neural Network.

        Args:
            layers (List): A list of layer instances.
            loss (Callable): A function that computes the loss.
            loss_derivative (Callable): A function that computes the derivative of the loss.

        Raises:
            TypeError: If layers is not a list.
        """
        if not isinstance(layers, list):
            raise TypeError("layers must be provided as a list of layer instances.")
        self.layers = layers
        self.loss = loss
        self.loss_derivative = loss_derivative

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: The network's output.

        Raises:
            ValueError: If X is not a numpy array.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data X must be a numpy array.")
        output = X
        try:
            for layer in self.layers:
                output = layer.forward(output)
        except Exception as e:
            raise RuntimeError(f"Error during forward propagation: {e}")
        return output

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Perform backward propagation through the network.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            output (np.ndarray): Output from the forward pass.

        Raises:
            ValueError: If input arrays are not numpy arrays.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(output, np.ndarray):
            raise ValueError("X, y, and output must be numpy arrays.")
        try:
            # Compute initial gradient from loss derivative
            grad = self.loss_derivative(y, output)
            # Propagate the gradient backward through the layers
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
        except Exception as e:
            raise RuntimeError(f"Error during backward propagation: {e}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, verbose: bool = True) -> None:
        """
        Train the neural network.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
            verbose (bool): If True, prints loss at each epoch.

        Raises:
            ValueError: If training data is not a numpy array or if epochs/learning_rate are not of correct type.
        """
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("Training data and labels must be numpy arrays.")
        if not isinstance(epochs, int) or not isinstance(learning_rate, (float, int)):
            raise ValueError("Epochs must be an integer and learning_rate must be a float or integer.")

        for epoch in range(epochs):
            output = self.forward(X_train)
            loss_value = self.loss(y_train, output)
            self.backward(X_train, y_train, output)
            # Update parameters in each layer that supports updates
            for layer in self.layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(learning_rate)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value}")
