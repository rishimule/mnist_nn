"""
layers.py

This module contains implementations of neural network layers.
Currently, it includes the Dense (fully connected) layer.
"""

from typing import Optional, Callable
import numpy as np
from src.neuralnet import activations

class Dense:
    """
    A fully connected (dense) neural network layer.

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of neurons in the layer.
        weights (np.ndarray): Weight matrix.
        biases (np.ndarray): Bias vector.
        activation (Optional[Callable]): Activation function.
        activation_derivative (Optional[Callable]): Derivative of the activation function.
        input (np.ndarray): Input data for the current forward pass.
        z (np.ndarray): Linear combination of input and weights plus bias.
        dweights (np.ndarray): Gradient of the weights.
        dbiases (np.ndarray): Gradient of the biases.
    """
    def __init__(self, input_size: int, output_size: int,
                 activation_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 activation_deriv: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """
        Initialize the Dense layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of neurons.
            activation_func (Optional[Callable]): Activation function to apply.
            activation_deriv (Optional[Callable]): Derivative of the activation function.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation_func
        self.activation_derivative = activation_deriv

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass through the layer.

        Args:
            input_data (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: Output after linear transformation and activation.

        Raises:
            ValueError: If input_data is not a numpy array.
        """
        if not isinstance(input_data, np.ndarray):
            raise ValueError("input_data must be a numpy array.")
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        try:
            if self.activation:
                self.output = self.activation(self.z)
            else:
                self.output = self.z
        except Exception as e:
            raise RuntimeError(f"Error applying activation function: {e}")
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass for this layer.

        Args:
            output_gradient (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input.

        Raises:
            ValueError: If output_gradient is not a numpy array.
        """
        if not isinstance(output_gradient, np.ndarray):
            raise ValueError("output_gradient must be a numpy array.")
        try:
            if self.activation_derivative:
                delta = output_gradient * self.activation_derivative(self.z)
            else:
                delta = output_gradient
            self.dweights = np.dot(self.input.T, delta)
            self.dbiases = np.sum(delta, axis=0, keepdims=True)
            input_gradient = np.dot(delta, self.weights.T)
        except Exception as e:
            raise RuntimeError(f"Error during backward pass in Dense layer: {e}")
        return input_gradient

    def update_params(self, learning_rate: float) -> None:
        """
        Update the layer's weights and biases using gradient descent.

        Args:
            learning_rate (float): Learning rate for the update.
        """
        try:
            self.weights -= learning_rate * self.dweights
            self.biases -= learning_rate * self.dbiases
        except Exception as e:
            raise RuntimeError(f"Error updating parameters in Dense layer: {e}")
