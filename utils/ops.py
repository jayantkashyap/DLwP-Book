import numpy as np


class Ops(object):

    def __init__(self):
        pass

    def der(self):
        pass


class Sigmoid(Ops):

    def __init__(self, z):
        self._z = np.array(z)

    def sigmoid(self):
        return 1/(1+np.exp(-self._z))

    def der(self):
        s = self.sigmoid()
        return s*(1-s)

    def __repr__(self):
        return 'sigmoid'


class Tanh(Ops):

    def __init__(self, z):
        self._z = np.array(z)

    def tanh(self):
        return (np.exp(self._z)-np.exp(-self._z))/(np.exp(self._z)+np.exp(-self._z))

    def der(self):
        t = self.sigmoid()
        return 1-(t**2)

    def __repr__(self):
        return 'tanh'


class Linear(Ops):

    def __init__(self, z):
        self._z = np.array(z)

    def linear(self):
        return self._z

    def der(self):
        return np.ones_like(self._z)

    def __repr__(self):
        return 'linear'


def softmax(x):
    s = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x)/s


