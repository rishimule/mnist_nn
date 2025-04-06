"""
activations.py

This module implements common activation functions and their derivatives.
"""

from typing import Any
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying sigmoid.

    Raises:
        ValueError: If x is not a numpy array.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy array.")
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function.

    Args:
        x (np.ndarray): Input array (pre-activation).

    Returns:
        np.ndarray: Derivative of the sigmoid function.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying ReLU.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy array.")
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU function.

    Args:
        x (np.ndarray): Input array (pre-activation).

    Returns:
        np.ndarray: Derivative of ReLU.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy array.")
    grad = (x > 0).astype(x.dtype)
    return grad

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Probabilities after applying softmax.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy array.")
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

# Note: The derivative of softmax is typically combined with the cross-entropy loss.
