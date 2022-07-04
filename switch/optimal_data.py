import numpy as np


def normalize(x):
    Test = x[:, 1:]
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x[:, 1:] = np.divide(np.subtract(Test, mu[1:]),
                         sigma[1:])
    return x


def add_row(x_train, counts: int):
    for count in range(counts):
        x_train1 = (x_train[:, (count + 3)] ** 2).reshape((len(x_train), 1))
        x_train = np.concatenate((x_train1, x_train), axis=1)
    return x_train


def add_bias(x):
    m = len(x)
    bias = np.ones(m)
    x = np.vstack((bias, x.T)).T
    return x
