import numpy as np 
from base import Loss

class MSE(Loss):
    def loss(self, predicted, actual):
        return np.sum((predicted - actual)**2)

    def grad(self, predicted, actual):
        return 2 * (predicted - actual)
