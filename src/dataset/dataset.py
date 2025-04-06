"""
dataset.py

This module provides functions to download, load, and preprocess the MNIST dataset.
It includes functions to read the raw IDX files, normalize the image data, and return
the training and testing sets.
"""

import os
import struct
import numpy as np
from typing import Tuple

def load_mnist_images(file_path: str) -> np.ndarray:
    """
    Load MNIST image data from the given IDX file.

    Args:
        file_path (str): Path to the IDX image file.

    Returns:
        np.ndarray: Numpy array of shape (num_images, rows, cols) containing the images.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError: If an error occurs during file reading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            # Read magic number and dimensions
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in MNIST image file: {file_path}")
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape((num_images, rows, cols))
    except Exception as e:
        raise RuntimeError(f"Error reading MNIST image file {file_path}: {e}")
    return images

def load_mnist_labels(file_path: str) -> np.ndarray:
    """
    Load MNIST label data from the given IDX file.

    Args:
        file_path (str): Path to the IDX label file.

    Returns:
        np.ndarray: Numpy array of shape (num_labels,) containing the labels.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError: If an error occurs during file reading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            # Read magic number and number of labels
            magic, num_labels = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in MNIST label file: {file_path}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Error reading MNIST label file {file_path}: {e}")
    return labels

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize and reshape image data.

    This function normalizes pixel values to the range [0, 1] and reshapes the images
    to have a shape of (num_images, rows * cols).

    Args:
        images (np.ndarray): Raw image data of shape (num_images, rows, cols).

    Returns:
        np.ndarray: Preprocessed image data of shape (num_images, rows * cols).

    Raises:
        ValueError: If input images is not a numpy array.
        RuntimeError: If an error occurs during preprocessing.
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Images must be a numpy array.")
    try:
        num_images = images.shape[0]
        # Normalize pixel values to [0, 1]
        images_normalized = images.astype(np.float32) / 255.0
        # Reshape images to 2D array: (num_images, rows*cols)
        images_reshaped = images_normalized.reshape(num_images, -1)
    except Exception as e:
        raise RuntimeError(f"Error preprocessing images: {e}")
    return images_reshaped

def load_and_preprocess_mnist(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the MNIST dataset from the specified directory.

    The function expects the following files to be present in data_dir:
      - train-images-idx3-ubyte
      - train-labels-idx1-ubyte
      - t10k-images-idx3-ubyte
      - t10k-labels-idx1-ubyte

    Args:
        data_dir (str): Directory containing the MNIST files.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
                - Preprocessed training images (numpy.ndarray)
                - Training labels (numpy.ndarray)
                - Preprocessed testing images (numpy.ndarray)
                - Testing labels (numpy.ndarray)

    Raises:
        RuntimeError: If an error occurs while loading or preprocessing the data.
    """
    try:
        train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
        train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
        test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
        test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

        train_images = load_mnist_images(train_images_path)
        train_labels = load_mnist_labels(train_labels_path)
        test_images = load_mnist_images(test_images_path)
        test_labels = load_mnist_labels(test_labels_path)

        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)
    except Exception as e:
        raise RuntimeError(f"Error loading and preprocessing MNIST dataset: {e}")

    return train_images, train_labels, test_images, test_labels
