import numpy as np
class GD_Momentum():
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    def apply_optimizer(self, param_name, value, gradient):
        if param_name not in self.velocities:
            self.velocities[param_name] = 0.0

        self.velocities[param_name] = self.momentum * self.velocities[param_name] - self.learning_rate * gradient
        value += self.velocities[param_name]
        return value  
