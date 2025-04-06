"""
network.py

This module defines the NeuralNetwork class which manages the network's layers,
forward and backward propagation, and training process.
"""

import numpy as np

class NeuralNetwork:
    """
    NeuralNetwork class to build and train a modular neural network.

    Attributes:
        layers (list): List of layers in the network.
        loss (function): Loss function to evaluate performance.
        loss_derivative (function): Derivative of the loss function.
    """
    def __init__(self, layers, loss, loss_derivative):
        """
        Initialize the Neural Network.

        Args:
            layers (list): A list of layer instances.
            loss (function): A function that computes the loss.
            loss_derivative (function): A function that computes the derivative of the loss.
        """
        self.layers = layers
        self.loss = loss
        self.loss_derivative = loss_derivative

    def forward(self, X):
        """
        Perform forward propagation through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: The network's output.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y, output):
        """
        Perform backward propagation through the network.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.
            output (numpy.ndarray): Output from the forward pass.
        """
        # Compute initial gradient from loss derivative
        grad = self.loss_derivative(y, output)
        # Propagate the gradient backward through the layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X_train, y_train, epochs, learning_rate, verbose=True):
        """
        Train the neural network.

        Args:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
            verbose (bool): If True, prints loss at each epoch.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)
            # Compute loss
            loss_value = self.loss(y_train, output)
            # Backward pass
            self.backward(X_train, y_train, output)
            # Update parameters in each layer that supports updates
            for layer in self.layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(learning_rate)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value}")
