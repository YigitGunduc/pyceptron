import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    y = tanh(x)
    return 1 - y**2
