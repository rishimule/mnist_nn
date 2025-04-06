"""
test_network.py

This module contains unit tests for various components of the neural network,
including layers, activation functions, loss functions, and the overall network
training process.
"""

import unittest
import numpy as np

from src.neuralnet.layers import Dense
from src.neuralnet.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax
from src.neuralnet.losses import mse_loss, mse_loss_derivative, cross_entropy_loss, cross_entropy_loss_derivative
from src.neuralnet.network import NeuralNetwork

class TestActivations(unittest.TestCase):
    def test_sigmoid(self) -> None:
        x = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid(x), expected, decimal=6)

    def test_sigmoid_derivative(self) -> None:
        x = np.array([-1, 0, 1])
        s = sigmoid(x)
        expected = s * (1 - s)
        np.testing.assert_array_almost_equal(sigmoid_derivative(x), expected, decimal=6)

    def test_relu(self) -> None:
        x = np.array([-1, 0, 1])
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_derivative(self) -> None:
        x = np.array([-1, 0, 1])
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(relu_derivative(x), expected)

    def test_softmax(self) -> None:
        x = np.array([[1, 2, 3]])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)
        self.assertEqual(result.shape, x.shape)

class TestLosses(unittest.TestCase):
    def test_mse_loss(self) -> None:
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertAlmostEqual(mse_loss(y_true, y_pred), 0.0)

    def test_mse_loss_derivative(self) -> None:
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        derivative = mse_loss_derivative(y_true, y_pred)
        expected = (y_pred - y_true) / y_true.shape[0]
        np.testing.assert_array_almost_equal(derivative, expected, decimal=6)

    def test_cross_entropy_loss(self) -> None:
        # Test with one sample: one-hot target and predicted probabilities.
        y_true = np.array([[0, 1]])
        y_pred = np.array([[0.25, 0.75]])
        loss_value = cross_entropy_loss(y_true, y_pred)
        expected = -np.log(0.75)
        self.assertAlmostEqual(loss_value, expected, places=6)

    def test_cross_entropy_loss_derivative(self) -> None:
        y_true = np.array([[0, 1]])
        y_pred = np.array([[0.25, 0.75]])
        derivative = cross_entropy_loss_derivative(y_true, y_pred)
        expected = (y_pred - y_true) / y_true.shape[0]
        np.testing.assert_array_almost_equal(derivative, expected, decimal=6)

class TestLayers(unittest.TestCase):
    def test_dense_forward(self) -> None:
        # Create a dense layer with fixed weights and biases.
        layer = Dense(2, 3, activation_func=None)
        layer.weights = np.array([[1, 2, 3], [4, 5, 6]])
        layer.biases = np.array([[1, 1, 1]])
        input_data = np.array([[1, 2]])
        # Expected: np.dot(input, weights) + biases = [10, 13, 16]
        expected = np.array([[10, 13, 16]])
        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected, decimal=6)

    def test_dense_backward(self) -> None:
        # Test backward pass for Dense layer.
        layer = Dense(2, 2, activation_func=None)
        layer.weights = np.array([[1, 2], [3, 4]])
        layer.biases = np.array([[0, 0]])
        input_data = np.array([[1, 1]])
        _ = layer.forward(input_data)
        output_gradient = np.array([[1, 1]])
        input_gradient = layer.backward(output_gradient)
        # Expected: output_gradient dot transpose(weights) = [3, 7]
        expected = np.array([[3, 7]])
        np.testing.assert_array_almost_equal(input_gradient, expected, decimal=6)


class TestNeuralNetwork(unittest.TestCase):
    def test_network_forward(self) -> None:
        # Build a simple network with one dense layer.
        layer = Dense(3, 2, activation_func=None)
        layer.weights = np.array([[1, 0], [0, 1], [1, 1]])
        layer.biases = np.array([[0, 0]])
        network = NeuralNetwork(
            layers=[layer],
            loss=lambda y, y_pred: np.mean((y - y_pred) ** 2),
            loss_derivative=lambda y, y_pred: (y_pred - y) / y.shape[0]
        )
        input_data = np.array([[1, 2, 3]])
        output = network.forward(input_data)
        expected = np.dot(input_data, layer.weights) + layer.biases
        np.testing.assert_array_almost_equal(output, expected, decimal=6)

    def test_training_step(self) -> None:
        # Test a single training step with a small dummy dataset.
        layer = Dense(4, 3, activation_func=softmax)
        network = NeuralNetwork(
            layers=[layer],
            loss=cross_entropy_loss,
            loss_derivative=cross_entropy_loss_derivative
        )
        X = np.array([[0.1, 0.2, 0.3, 0.4]])
        y = np.array([[0, 1, 0]])
        output_before = network.forward(X)
        loss_before = cross_entropy_loss(y, output_before)
        network.backward(X, y, output_before)
        for layer in network.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(0.1)
        output_after = network.forward(X)
        loss_after = cross_entropy_loss(y, output_after)
        # Verify that the loss decreases after one training step.
        self.assertLess(loss_after, loss_before)

if __name__ == "__main__":
    unittest.main()
