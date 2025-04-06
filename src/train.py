"""
train.py

This script implements the training pipeline for the MNIST neural network.
It loads and preprocesses the dataset, initializes the network, and runs the training loop with periodic
evaluation of the network’s performance on a validation set.
"""

import numpy as np
from typing import Tuple

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.dataset import load_and_preprocess_mnist
from src.neuralnet.network import NeuralNetwork
from src.neuralnet.layers import Dense
from src.neuralnet.activations import relu, relu_derivative, softmax
from src.neuralnet.losses import cross_entropy_loss, cross_entropy_loss_derivative
from src.neuralnet.utils import one_hot_encode

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true (np.ndarray): True labels (as integers).
        y_pred (np.ndarray): Predicted probabilities from the network.

    Returns:
        float: Accuracy as a percentage.
    """
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_true) * 100
    return accuracy

def train_network(train_images: np.ndarray, train_labels: np.ndarray,
                  val_images: np.ndarray, val_labels: np.ndarray,
                  epochs: int, learning_rate: float) -> NeuralNetwork:
    """
    Train the neural network on the training data and evaluate on the validation data.

    Args:
        train_images (np.ndarray): Training images (preprocessed).
        train_labels (np.ndarray): One-hot encoded training labels.
        val_images (np.ndarray): Validation images (preprocessed).
        val_labels (np.ndarray): One-hot encoded validation labels.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for updates.

    Returns:
        NeuralNetwork: The trained neural network.
    """
    # Define network architecture:
    # - Input layer size is determined by the flattened image size (28x28 = 784)
    # - Hidden layer: 128 neurons with ReLU activation
    # - Output layer: 10 neurons with softmax activation for 10 classes
    try:
        layer1 = Dense(784, 128, activation_func=relu, activation_deriv=relu_derivative)
        layer2 = Dense(128, 10, activation_func=softmax, activation_deriv=None)
    except Exception as e:
        raise RuntimeError(f"Error initializing layers: {e}")

    network = NeuralNetwork(
        layers=[layer1, layer2],
        loss=cross_entropy_loss,
        loss_derivative=cross_entropy_loss_derivative
    )

    # Training loop
    for epoch in range(epochs):
        try:
            # Forward pass on training data
            train_output = network.forward(train_images)
            train_loss = cross_entropy_loss(train_labels, train_output)
            # Backward pass and parameter update
            network.backward(train_images, train_labels, train_output)
            for layer in network.layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(learning_rate)
        except Exception as e:
            raise RuntimeError(f"Error during training at epoch {epoch+1}: {e}")

        # Periodic validation evaluation (evaluated every epoch here; adjust as needed)
        try:
            val_output = network.forward(val_images)
            val_loss = cross_entropy_loss(val_labels, val_output)
            val_accuracy = compute_accuracy(np.argmax(val_labels, axis=1), val_output)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        except Exception as e:
            raise RuntimeError(f"Error during validation evaluation at epoch {epoch+1}: {e}")

    return network

def main() -> None:
    """
    Main function to run the training pipeline.
    """
    try:
        # Load and preprocess the MNIST dataset
        data_dir = "data/mnist"
        train_images_all, train_labels_all, test_images, test_labels = load_and_preprocess_mnist(data_dir)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

    # Split the training data into training and validation sets (e.g., 80% training, 20% validation)
    split_index = int(0.8 * train_images_all.shape[0])
    train_images = train_images_all[:split_index]
    train_labels = train_labels_all[:split_index]
    val_images = train_images_all[split_index:]
    val_labels = train_labels_all[split_index:]

    # Convert integer labels to one-hot encoding (required by the cross-entropy loss)
    num_classes = 10
    train_labels_encoded = one_hot_encode(train_labels, num_classes)
    val_labels_encoded = one_hot_encode(val_labels, num_classes)
    test_labels_encoded = one_hot_encode(test_labels, num_classes)

    # Set training hyperparameters
    epochs = 40
    learning_rate = 0.01

    # Train the network
    trained_network = train_network(train_images, train_labels_encoded, val_images, val_labels_encoded, epochs, learning_rate)

    # Evaluate the trained network on the test set
    try:
        test_output = trained_network.forward(test_images)
        test_loss = cross_entropy_loss(test_labels_encoded, test_output)
        test_accuracy = compute_accuracy(test_labels, test_output)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    except Exception as e:
        raise RuntimeError(f"Error during test evaluation: {e}")

if __name__ == "__main__":
    main()
