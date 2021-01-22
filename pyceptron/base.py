class Layer:
    def __init__(self):
        self.w, self.b = None, None
        self.dw, self.db = None, None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Activation(Layer):
    def __init__(self, f, f_prime):
        super().__init__()
        self.layerType = 'activation'
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        return self.f_prime(self.inputs) * grad

class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError

