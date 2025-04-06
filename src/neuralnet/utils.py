"""
utils.py

This module includes utility functions for the neural network, including weight initialization,
data normalization, and label preprocessing.
"""

from typing import Tuple
import numpy as np

def initialize_weights(shape: Tuple[int, int], scale: float = 0.01) -> np.ndarray:
    """
    Initialize weights with small random values.

    Args:
        shape (Tuple[int, int]): Shape of the weight matrix.
        scale (float): Scaling factor for the random values.

    Returns:
        np.ndarray: Initialized weights.
    """
    try:
        return np.random.randn(*shape) * scale
    except Exception as e:
        raise RuntimeError(f"Error initializing weights with shape {shape}: {e}")

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to have values between 0 and 1.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Normalized data.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    try:
        return data / 255.0
    except Exception as e:
        raise RuntimeError(f"Error normalizing data: {e}")

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode label data.

    Args:
        labels (np.ndarray): Array of labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded labels.
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError("Labels must be a numpy array.")
    try:
        encoded = np.zeros((labels.size, num_classes))
        encoded[np.arange(labels.size), labels] = 1
        return encoded
    except Exception as e:
        raise RuntimeError(f"Error in one-hot encoding: {e}")
