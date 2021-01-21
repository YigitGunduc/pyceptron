import numpy as np
from base import Layer, Activation
from activations import tanh, tanh_prime

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
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


