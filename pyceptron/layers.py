import numpy as np
from pyceptron.base import Layer, Activation
from pyceptron.activations import tanh, tanh_prime, sigmoid, sigmoid_prime

class Linear(Layer):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.w = np.random.randn(input_size, hidden_units)
        self.b = np.random.randn(hidden_units)
        self.layerType = 'linear'

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.w + self.b

    def backward(self, grad):
        self.db = np.sum(grad, axis=0)
        self.dw = self.inputs.T @ grad
        return grad @ self.w.T

    def get_weights(self):
        return self.w, self.b

    def get_grads(self):
        if self.dw is not None or self.db is not None:
            pass
        return self.dw, self.db

    def setWeigths(self, w, b):
        self.w = w
        self.b = b

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class ReluLayer(Layer):
    def __init__(self):
        self._z = None

    def forward(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._z = np.maximum(0, a_prev)
        return self._z

    def backward(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz