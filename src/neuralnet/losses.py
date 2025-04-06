"""
losses.py

This module implements common loss functions and their derivatives.
"""

import numpy as np

def mse_loss(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) loss.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted outputs.

    Returns:
        float: The MSE loss.
    """
    return np.mean(np.power(y_true - y_pred, 2)) / 2

def mse_loss_derivative(y_true, y_pred):
    """
    Compute the derivative of the MSE loss.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted outputs.

    Returns:
        numpy.ndarray: Derivative of the MSE loss.
    """
    return (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    Compute the cross-entropy loss.

    Args:
        y_true (numpy.ndarray): True labels (one-hot encoded).
        y_pred (numpy.ndarray): Predicted probabilities.
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: The cross-entropy loss.
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_loss_derivative(y_true, y_pred):
    """
    Compute the derivative of the cross-entropy loss.
    Note: When using softmax activation in the output layer, this derivative simplifies to (y_pred - y_true).

    Args:
        y_true (numpy.ndarray): True labels (one-hot encoded).
        y_pred (numpy.ndarray): Predicted probabilities.

    Returns:
        numpy.ndarray: Derivative of the cross-entropy loss.
    """
    return (y_pred - y_true) / y_true.shape[0]
