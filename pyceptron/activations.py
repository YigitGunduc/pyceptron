import numpy as np


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    y = tanh(x)
    return 1 - y**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(-x))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    dz = np.array(x, copy=True)
    dz[z <= 0] = 0
    return dz

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def softmax_prime(x):
    return x

