import numpy as np


class CategoricalCrossEntropy(): 
    def __init__(self, name):
        self.name = name
    def cross_entropy_loss(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.batch_size = targets.shape[0]
        return -np.sum(targets * np.log(predictions + 1**-100))/self.batch_size

    def cross_entropy_loss_prime(self):
        de_dx_store = np.zeros(shape=(self.targets.shape))
        for i in range(self.targets.shape[0]):
            de_dx_store[i] = -self.targets[i]/(self.predictions[i]+ 1**-100)
        return np.mean(de_dx_store, axis=0)



'''
c = CategoricalCrossEntropy()
predictions = np.array([[0.2, 0.3, 0.24, 0.9],[0.8, 0.11, 0.06, 0.3]])
targets = np.array([[0,0,0,1],[1,0,0,0]])
print(c.cross_entropy_loss(predictions, targets))
print(c.cross_entropy_loss_prime())
'''

