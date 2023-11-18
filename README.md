# Bee vs. Wasp Image Classifier

## Overview

This repository contains a Convolutional Neural Network (CNN) model trained to classify images as either a bee or a wasp. The model was trained using TensorFlow and Keras, and the architecture is described in the code. The training accuracy on the test dataset reached approximately 80%.

## Model Architecture

The CNN model follows a simple architecture:

1. Input shape: (150, 150, 3)
2. Convolutional layer with 32 filters, kernel size (3, 3), and 'relu' activation
3. MaxPooling layer with pool size (2, 2)
4. Flatten layer to convert the output to vectors
5. Dense layer with 64 neurons and 'relu' activation
6. Output layer with 1 neuron and 'sigmoid' activation for binary classification (bee or wasp)

## Training

The model was trained using a Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.002 and momentum of 0.8. The training dataset included images of both bees and wasps, and the validation dataset was used to evaluate the model's performance during training.

## Evaluation

The model achieved an accuracy of approximately 80% on the test dataset, indicating its ability to distinguish between bee and wasp images. The accuracy metric is a measure of how well the model performs on unseen data.

## How to Use

To use the trained model for predictions on new images, follow these steps:

1. Load the trained model using the provided code.
2. Preprocess the input image to match the model's input shape (150x150 pixels, 3 channels).
3. Use the model to make predictions on the preprocessed image.
4. Interpret the model's output - a probability score close to 1 indicates the image is likely a bee, while a score close to 0 indicates a wasp.

## Dependencies

- TensorFlow
- Keras
- Numpy
