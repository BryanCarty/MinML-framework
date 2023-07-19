
import numpy as np
import sys

class ValidMaxPooling:
    def __init__(self, pool_size, stride, name):
        self.pool_size = pool_size
        self.stride = stride
        self.name = name
    
    def forward(self, input_array): # batch, channels, height, width
        self.input = input_array
        self.batch_size, self.channels, input_height, input_width = input_array.shape
        self.output_height = (input_height - self.pool_size) // self.stride + 1
        self.output_width = (input_width - self.pool_size) // self.stride + 1
        pooled_array = np.zeros(shape=(self.batch_size, self.channels, self.output_height, self.output_width))
        self.gradient_indexes = np.zeros(shape=(self.batch_size*self.channels*self.output_height*self.output_width, 4))
        z = 0
        for b in range(self.batch_size):
            for c in range(self.channels):    
                m = self.input[b][c]
                for i in range(self.output_height): # rows
                    for j in range(self.output_width): # columns
                        patch = m[i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                        pooled_array[b,c,i,j] = np.max(patch)
                        max_index_in_patch = np.unravel_index(np.argmax(patch, axis=None), patch.shape)
                        max_index_in_input = np.array([b, c,int(i*self.stride + max_index_in_patch[0]),int(j*self.stride + max_index_in_patch[1])])
                        self.gradient_indexes[z] = max_index_in_input
                        z+=1                 
        return pooled_array
    
    def backward(self, de_dy):
        de_dy = de_dy.flatten()
        de_dy_len = len(de_dy)
        de_dx_store = np.zeros(shape=self.input.shape)
        
        for i in range(len(self.gradient_indexes)): #120960
            a, b, c, d = self.gradient_indexes[i]
            de_dx_store[int(a),int(b),int(c),int(d)]+=de_dy[i%de_dy_len]
        
        return np.mean(de_dx_store, axis=0)





'''
shape = (5,3,7,7)  # Example shape: 3 rows, 4 columns
# Generate a random array with values ranging from -10 to +10
arr = np.random.randint(-10, 11, size=shape)
mp = ValidMaxPooling(3,2)
mp.forward(arr)
print(mp.gradient_indexes)


print(arr)
print('-------------------')
mp = ValidMaxPooling(3,2)
print(mp.forward(arr))
print('------------------')
print(mp.gradients)
'''
