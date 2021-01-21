import numpy as np

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

    def add(self,layer):
        self.layers.append(layer)

    def step(self):
        for layer in self.layers:
            if layer.layerType is not 'activation':
                w, b = layer.get_weights()
                dw, db = layer.get_grads()
                w -= 0.01 * dw
                b -= 0.01 * db
                layer.setWeigths(w, b)

    def fit(self, inputs, targets, loss):
        for epoch in range(5000):
            epoch_loss = 0.0
            predicted = self.forward(inputs)
            epoch_loss += loss.loss(predicted, targets)
            grad = loss.grad(predicted, targets)
            self.backward(grad)
            self.step()
            print(f"{epoch} {epoch_loss:.9f}")