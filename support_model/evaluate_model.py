import numpy as np


def calculate_loss(y, y_predict):
    return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))


def accuracy(y, y_predict):
    for value in range(len(y_predict)):
        if y_predict[value] > 0.5:
            y_predict[value] = 1
        else:
            y_predict[value] = 0
    accuracy_y = np.mean(y_predict == y) * 100
    return accuracy_y
