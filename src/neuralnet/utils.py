"""
utils.py

This module includes utility functions for the neural network, including weight initialization,
data normalization, and label preprocessing.
"""

import numpy as np

def initialize_weights(shape, scale=0.01):
    """
    Initialize weights with small random values.

    Args:
        shape (tuple): Shape of the weight matrix.
        scale (float): Scaling factor for the random values.

    Returns:
        numpy.ndarray: Initialized weights.
    """
    return np.random.randn(*shape) * scale

def normalize_data(data):
    """
    Normalize data to have values between 0 and 1.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Normalized data.
    """
    return data / 255.0

def one_hot_encode(labels, num_classes):
    """
    One-hot encode label data.

    Args:
        labels (numpy.ndarray): Array of labels.
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: One-hot encoded labels.
    """
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded
