import numpy as np
import sys
class Dense():
    def __init__(self, input_neurons, output_neurons, optimizer, name):
        self.name = name
        self.input_neurons = input_neurons
        self.weights = np.random.normal(loc=0, scale=0.01, size=(input_neurons, output_neurons))
        self.biases = np.zeros(shape=(output_neurons))
        self.output_neurons = output_neurons
        self.optimizer = optimizer
    def forward(self, input):
        self.input = input
        output = np.zeros(shape=(len(self.input), self.output_neurons))
        for i in range(len(self.input)):
            output[i] = np.dot(self.weights.T, self.input[i])+self.biases
        return output
    def backward(self, de_dy):
        # Need to calculate de_dw, de_db, de_dx & also update weights and biases
        # de_dw = de_dy * dy_dw
        # de_db = de_dy * dy_db
        # de_dx = de_dy * dy_dx
        de_dw_store = np.zeros(shape=(len(self.input), *self.weights.shape))
        
        de_dy = de_dy.reshape(self.output_neurons,1) # 8,1
        dy_dw = self.input.reshape(len(self.input),self.input_neurons,1) # (128,475,1)
        dy_dx = self.weights # (475,8)

        de_dx = np.dot(de_dy.T, dy_dx.T).reshape(self.input_neurons,) # (1,8) x (8,475) = (1,475)
        de_db = de_dy.reshape(self.output_neurons,)

        for i in range(len(self.input)): # 128, 475
            de_dw = np.dot(de_dy, dy_dw[i].T).T # (8,1) x (1,475) = (8,475) = (475,8)
            de_dw_store[i] = de_dw

        avg_de_dw = np.mean(de_dw_store, axis=0)
        self.weights = self.optimizer.apply_optimizer(self.name+":W",self.weights, avg_de_dw)
        self.biases = self.optimizer.apply_optimizer(self.name+":B",self.biases, de_db)
        return de_dx



'''
dl = Dense(4, 2)
print(dl.weights)
print(dl.biases)

input = np.array([1,2,3,4])

print(dl.forward(input))
'''