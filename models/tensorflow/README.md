# TensorFlow CTR Prediction Framework

This directory contains the implementation of various CTR prediction models in TensorFlow.

The framework is designed to be modular and extensible, allowing for easy implementation and experimentation with different model architectures.

## Structure

-   `layers/`: Contains common TensorFlow layers used across different models.
-   `[model_name]/`: Contains the implementation of a specific CTR model.
-   `input_layer.py`: Handles input feature processing and embedding lookups.

## Getting Started

To use a specific model, import the model class from its respective directory.
