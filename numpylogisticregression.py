#!/usr/bin/python
# -*- coding: utf-8 -*-
# Put in Script
import numpy as np

def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))


class LogisticRegressionNumpy:

    def __init__(
        self,
        learning_rate=0.001,
        num_features=784,
        num_classes=10,
        y_true_length=None
        ):
    
        self.m = y_true_length
        self.learning_rate = learning_rate
        self.weights = np.zeros(shape=(784, 1))
        self.bias = np.zeros(shape=(1, 10))

    def loss_fn(self, y_true, y_pred):
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true)
                        * np.log(1 - y_pred))
        return cost

    def predict(self, batch_features, batch_labels):
        theta = np.matmul(batch_features, self.weights) + self.bias
        prediction = sigmoid(theta)
        loss = self.loss_fn(batch_labels, prediction)

    # calculate the derivative of theta, weights and bias

        dTheta = prediction - batch_labels
        dWeights = 1 / self.m * np.matmul(batch_features.T, dTheta)
        dBias = np.sum(dTheta)

    # apply the gradient descent

        self.weights = self.weights - self.learning_rate * dWeights
        self.bias = self.bias - self.learning_rate * dBias

        return loss



			