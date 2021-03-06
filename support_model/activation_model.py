import numpy as np

"""
    Activation Function of Model
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)


def sigmoid_derivative(x):
    return x * (1 - x)


def relu_derivative(x):
    return x > 0
