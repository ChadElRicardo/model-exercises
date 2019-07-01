#!/usr/bin/python
# -*- coding: utf-8 -*-
# Put in Script

import numpy as np


class Layer:

    def __init__(self):

    # input shape is 784

        self.weights = np.zeros(shape=(784, 10))
        bias = np.zeros(shape=(10, ))

    def forward(self, input):
        output = np.matmul(input, self.weights) + bias
        return output


class Dense(Layer):

    def __init__(
        self,
        input_units,
        output_units,
        learning_rate=0.1,
        ):
        self.learning_rate = learning_rate

    # Initialize weights with small numbers

        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.matmul(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, np.transpose(self.weights))

        grad_weights = np.transpose(np.dot(np.transpose(grad_output),
                                    input))
        grad_biases = np.sum(grad_output, axis=0)

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


class ReLU(Layer):

    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad

			