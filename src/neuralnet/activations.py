"""
activations.py

This module implements common activation functions and their derivatives.
"""

import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output after applying sigmoid.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function.

    Args:
        x (numpy.ndarray): Input array (pre-activation).

    Returns:
        numpy.ndarray: Derivative of the sigmoid function.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    Compute the ReLU activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output after applying ReLU.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Compute the derivative of the ReLU function.

    Args:
        x (numpy.ndarray): Input array (pre-activation).

    Returns:
        numpy.ndarray: Derivative of ReLU.
    """
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def softmax(x):
    """
    Compute the softmax activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Probabilities after applying softmax.
    """
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

# Note: The derivative of softmax is usually computed in combination with the cross-entropy loss.
