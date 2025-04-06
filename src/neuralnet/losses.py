"""
losses.py

This module implements common loss functions and their derivatives.
"""

from typing import Any
import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) loss.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted outputs.

    Returns:
        float: The MSE loss.
    """
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise ValueError("y_true and y_pred must be numpy arrays.")
    return float(np.mean(np.power(y_true - y_pred, 2)) / 2)

def mse_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the MSE loss.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted outputs.

    Returns:
        np.ndarray: Derivative of the MSE loss.
    """
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise ValueError("y_true and y_pred must be numpy arrays.")
    return (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute the cross-entropy loss.

    Args:
        y_true (np.ndarray): True labels (one-hot encoded).
        y_pred (np.ndarray): Predicted probabilities.
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: The cross-entropy loss.
    """
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise ValueError("y_true and y_pred must be numpy arrays.")
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

def cross_entropy_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the cross-entropy loss.
    Note: When using softmax activation in the output layer, this derivative simplifies to (y_pred - y_true).

    Args:
        y_true (np.ndarray): True labels (one-hot encoded).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Derivative of the cross-entropy loss.
    """
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise ValueError("y_true and y_pred must be numpy arrays.")
    return (y_pred - y_true) / y_true.shape[0]
