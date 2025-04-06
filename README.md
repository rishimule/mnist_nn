# MNIST Neural Network Project

This project builds a neural network from scratch in Python to recognize handwritten digits from the MNIST dataset. The code is organized in a modular fashion, making it easy to reuse and extend in future projects.

## Features

- **Neural Network from Scratch:** Implemented without using any machine learning libraries.
- **Modular Architecture:** Separated modules for dataset handling, network architecture, layers, activations, loss functions, optimizers, and utilities.
- **Training Pipeline:** Includes forward and backward propagation, loss computation, weight updates, and validation.
- **Testing:** Unit and integration tests ensure reliability of individual components and the overall training process.
- **Jupyter Notebook Integration:** Interactive notebook for training, visualization, and analysis of results.
- **Extensibility:** Well-documented and structured for future enhancements and integration into other projects.

## Project Structure

```plaintext
mnist_nn/
├── README.md                # Project overview and instructions (this file)
├── LICENSE                  # License information
├── requirements.txt         # List of required Python libraries and their versions
├── data/
│   └── mnist/               # MNIST dataset files (download and extract here)
├── docs/
│   └── design_documentation.md  # Detailed design decisions and future extension plans
├── notebooks/
│   └── mnist_nn.ipynb       # Jupyter Notebook for interactive training and visualization
└── src/
    ├── dataset/             # Module for loading and preprocessing the MNIST dataset
    │   └── dataset.py
    ├── neuralnet/           # Neural network core module
    │   ├── __init__.py
    │   ├── network.py       # NeuralNetwork class for training and inference
    │   ├── layers.py        # Dense layer implementation and others
    │   ├── activations.py   # Activation functions and their derivatives
    │   ├── losses.py        # Loss functions and gradients (e.g., cross-entropy)
    │   ├── optimizers.py    # Simple optimizers (e.g., gradient descent)
    │   └── utils.py         # Utility functions (e.g., weight initialization, one-hot encoding)
    └── tests/               # Unit and integration tests
        └── test_network.py
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rishimule/mnist_nn.git
   cd mnist_nn
   ```

2. **Set Up the Conda Environment:**

   Create a new conda environment with Python 3.11 and install the required dependencies:

   ```bash
   conda create -n mnist_nn python=3.11
   conda activate mnist_nn
   pip install -r requirements.txt
   ```

3. **Download the MNIST Dataset:**

   Navigate to the `data/mnist` directory and run the following commands to download and extract the dataset:

   ```bash
   cd data/mnist
   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   gunzip *.gz
   cd ../../
   ```
   or
   ```bash
   cd data/mnist
   wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
   wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
   wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
   wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
   gunzip *.gz
   cd ../../
   ```

## Usage

### Training the Network

You can train the network using the provided training script:

```bash
python -m src.train
```

This script will load the MNIST dataset, initialize the network architecture, run the training loop, and evaluate the model on a test set.

### Using the Jupyter Notebook

To run the interactive Jupyter Notebook:

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `notebooks/mnist_nn.ipynb` to view and run the notebook cells, which include dataset loading, model configuration, training, and visualization of results.

## Running Tests

To run the unit and integration tests, execute the following command from the project root:

```bash
python -m src.tests.test_network
```

This will run tests for activation functions, loss functions, layer operations, and the overall training pipeline.

## Future Extensions

- **Additional Layer Types:** Extend `layers.py` to include convolutional, pooling, or recurrent layers.
- **Advanced Optimizers:** Implement optimizers like Adam or RMSProp in `optimizers.py`.
- **Enhanced Data Handling:** Incorporate data augmentation and advanced preprocessing techniques.
- **Dynamic Configuration:** Enable dynamic network configuration using external configuration files or command-line arguments.
- **Integration:** The modular design allows easy integration of this codebase into larger projects or frameworks.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## Acknowledgements

- [MNIST Dataset (lecun)](http://yann.lecun.com/exdb/mnist/), [2 (stackOverfow)](https://stackoverflow.com/a/66820249)
- This project was developed as an educational tool to demonstrate building neural networks from scratch.
```
