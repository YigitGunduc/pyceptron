import time as t
import numpy as np
from pyceptron.helpers import progressbar

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def append(self,layer):
        self.layers.append(layer)

    def fit(self, x=None, y=None, epochs=1, loss=None, optimizer=None):
        for epoch in progressbar(range(epochs), "Training: "):
            epoch_loss = 0.0
            predicted = self.forward(x)
            epoch_loss += loss.loss(predicted, y)
            grad = loss.grad(predicted, y)
            self.backward(grad)
            optimizer.step(self.layers)
        print(f'Loss: {epoch_loss}')



