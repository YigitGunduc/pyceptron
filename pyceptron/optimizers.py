import numpy as np


class Optimizer:
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr = 0.0001):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if layer.layerType != 'activation':
                w, b = layer.get_weights()
                dw, db = layer.get_grads()
                w -= 0.01 * dw
                b -= 0.01 * db
                layer.setWeigths(w, b)
