
import numpy as np
class ReLU():
    def __init__(self, name):
        self.name = name
    def forward(self, input):
        self.input = input
        return np.maximum(0, self.input)
    def backward(self, de_dy):
        de_dx_store = np.zeros(shape=self.input.shape)
        for i in range(len(self.input)):
            de_dx_store[i] = np.where(self.input[i] >= 0, de_dy, 0)
        de_dx = np.mean(de_dx_store, axis=0)
        return de_dx