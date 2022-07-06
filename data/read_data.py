import numpy as np

from switch.optimal_data import add_row, normalize

"""
Convert Data to training model
"""


def read_data(data: np):
    n, d = data.shape
    read_x = data[:, 0:d - 1].reshape(-1, d - 1)
    read_y = data[:, -1].reshape(-1, 1)
    return read_x, read_y


def convert_data(x: np, number):
    x_convert = add_row(x, number)
    x_convert = normalize(x_convert)
    return x_convert
