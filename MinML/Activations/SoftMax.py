import numpy as np
import sys

class Softmax:
    def __init__(self, name):
        self.name = name
    def forward(self, input):
        self.input = input
        self.output = np.zeros(shape=(input.shape[0], input.shape[1]))
        # Subtracting the maximum value for numerical stability
        for i in range(input.shape[0]):
            exp_vals = np.exp(input[i] - np.max(input[i], keepdims=True))
            self.output[i] = exp_vals / np.sum(exp_vals, keepdims=True)

        return self.output
    '''
    def backward(self, de_dy):
        # Calculate the gradient of the softmax layer
        de_dx_store = np.zeros(shape=(self.input.shape))
        for i in range(self.input.shape[0]):
            n = np.size(self.output[i])
            tmp = np.tile(self.output[i], n)
            de_dx_store[i] = np.dot(tmp * (np.identity(n) - np.transpose(tmp)), de_dy)
        return np.mean(de_dx_store, axis=0)
    '''
    
    def backward(self, de_dy):
        # Calculate the gradient of the softmax layer
        de_dx_store = np.zeros_like(self.input)
        for i in range(self.input.shape[0]):
            n = np.size(self.input[i])
            tmp = np.tile(self.output[i], (n, 1))
            diag = np.diag(self.output[i])
            jacobian = np.dot(tmp.T, (diag - np.dot(tmp, tmp.T)))
            de_dx_store[i] = np.dot(de_dy, jacobian)
        return np.mean(de_dx_store, axis=0)
    
