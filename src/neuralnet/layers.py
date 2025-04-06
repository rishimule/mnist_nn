"""
layers.py

This module contains implementations of neural network layers.
Currently, it includes the Dense (fully connected) layer.
"""

import numpy as np
from src.neuralnet import activations

class Dense:
    """
    A fully connected (dense) neural network layer.

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of neurons in the layer.
        weights (numpy.ndarray): Weight matrix.
        biases (numpy.ndarray): Bias vector.
        activation (function): Activation function.
        activation_derivative (function): Derivative of the activation function.
        input (numpy.ndarray): Input data for the current forward pass.
        z (numpy.ndarray): Linear combination of input and weights plus bias.
        dweights (numpy.ndarray): Gradient of the weights.
        dbiases (numpy.ndarray): Gradient of the biases.
    """
    def __init__(self, input_size, output_size, activation_func=None, activation_deriv=None):
        """
        Initialize the Dense layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of neurons.
            activation_func (function, optional): Activation function to apply.
            activation_deriv (function, optional): Derivative of the activation function.
        """
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights with small random values and biases with zeros.
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation_func
        self.activation_derivative = activation_deriv

    def forward(self, input_data):
        """
        Compute the forward pass through the layer.

        Args:
            input_data (numpy.ndarray): Input data to the layer.

        Returns:
            numpy.ndarray: Output after linear transformation and activation.
        """
        self.input = input_data
        # Linear transformation
        self.z = np.dot(self.input, self.weights) + self.biases
        # Apply activation function if provided
        if self.activation:
            self.output = self.activation(self.z)
        else:
            self.output = self.z
        return self.output

    def backward(self, output_gradient):
        """
        Compute the backward pass for this layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        # Compute gradient of the activation if applicable
        if self.activation_derivative:
            delta = output_gradient * self.activation_derivative(self.z)
        else:
            delta = output_gradient

        # Calculate gradients for weights and biases
        self.dweights = np.dot(self.input.T, delta)
        self.dbiases = np.sum(delta, axis=0, keepdims=True)
        # Calculate gradient to pass to previous layer
        input_gradient = np.dot(delta, self.weights.T)
        return input_gradient

    def update_params(self, learning_rate):
        """
        Update the layer's weights and biases using gradient descent.

        Args:
            learning_rate (float): Learning rate for the update.
        """
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
