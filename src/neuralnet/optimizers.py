"""
optimizers.py

This module implements optimizers for updating network parameters.
Currently, it includes a simple Gradient Descent optimizer.
"""

from typing import List
import numpy as np

class GradientDescent:
    """
    Simple Gradient Descent optimizer.
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initialize the optimizer.

        Args:
            learning_rate (float): Learning rate for parameter updates.
        """
        self.learning_rate = learning_rate

    def update(self, layers: List) -> None:
        """
        Update the parameters of each layer that supports updates.

        Args:
            layers (List): List of layers in the network.
        """
        try:
            for layer in layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(self.learning_rate)
        except Exception as e:
            raise RuntimeError(f"Error during parameter update: {e}")
