"""
optimizers.py

This module implements optimizers for updating network parameters.
Currently, it includes a simple Gradient Descent optimizer.
"""

class GradientDescent:
    """
    Simple Gradient Descent optimizer.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initialize the optimizer.

        Args:
            learning_rate (float): Learning rate for parameter updates.
        """
        self.learning_rate = learning_rate

    def update(self, layers):
        """
        Update the parameters of each layer that supports updates.

        Args:
            layers (list): List of layers in the network.
        """
        for layer in layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(self.learning_rate)
