import numpy as np
from scipy.signal import correlate2d, convolve2d
import sys
class Conv():
    def __init__(self, input_shape, kernel_shape, num_kernels, stride, optimizer, conv_mode, kernel_initializer, name, input_layer=False):
        self.name = name
        self.input_layer = input_layer
        self.conv_mode = conv_mode
        self.stride = stride
        self.optimizer = optimizer
        self.num_kernels = num_kernels
        self.batch_size, self.input_height, self.input_width, self.input_depth = input_shape
        if kernel_initializer == "he_normal":
            self.kernels = np.random.randn(*(num_kernels, self.input_depth, kernel_shape[0], kernel_shape[1])) * np.sqrt(2 / (self.input_depth * kernel_shape[0] * kernel_shape[1]))
        else:
            self.kernels = np.random.normal(loc=0, scale=0.01, size=(num_kernels, self.input_depth, kernel_shape[0], kernel_shape[1]))
        if conv_mode=="valid":
            self.biases = np.zeros(shape=(num_kernels, int((self.input_height-kernel_shape[0])/self.stride+1),int((self.input_width-kernel_shape[1])/self.stride+1)))
        elif conv_mode=="same":
            self.biases = np.zeros(shape=(num_kernels, int(np.ceil(self.input_height/self.stride)), int(np.ceil(self.input_width/self.stride))))
    def forward(self, input):
        self.input = np.array(input)
        batch_output = np.zeros((self.batch_size, *self.biases.shape))
        for b in range(self.batch_size): # iterates through elements in batch (128)
            instance_output = self.biases # bias for each kernel
            for k in range(self.num_kernels): # 40
                for i in range(self.input_depth): # 3
                    val = correlate2d(self.input[b][i], self.kernels[k][i], mode=self.conv_mode)
                    instance_output[k] += val[::self.stride, ::self.stride]
            batch_output[b] = instance_output
            self.output = batch_output
        return batch_output
    
    def backward(self, de_dy):
        de_db = de_dy # (105, 7, 7)
        de_dk_store = np.zeros(shape=(len(self.input),*self.kernels.shape)) 
        de_dx_store = np.zeros(shape=self.input.shape) # (128, 158, 8, 8)

        if self.input_layer == True: # Hardcoded to deal with first layer
            for b in range(self.batch_size):  # 128
                for k in range(self.num_kernels): # 105
                    for i in range(self.kernels.shape[1]): # 158
                        dilated_matrix = np.zeros(((2 * de_dy[k].shape[0])-1, (2 * de_dy[k].shape[1]-1)))
                        dilated_matrix[::2, ::2] = de_dy[k] 
                        de_dk_store[b,k,i] = correlate2d(self.input[b, i], dilated_matrix, "valid")
            de_dk_avg = np.mean(de_dk_store, axis=0)
            self.kernels = self.optimizer.apply_optimizer(self.name+":K", self.kernels, de_dk_avg)
            self.biases = self.optimizer.apply_optimizer(self.name+":B", self.biases, de_db)
            return

        for b in range(self.batch_size):  # 128
            for k in range(self.num_kernels): # 105
                for i in range(self.kernels.shape[1]): # 158
                    de_dk_store[b,k,i] = correlate2d(self.input[b, i], de_dy[k], "valid")
                    de_dx_store[b,i]+= convolve2d(de_dy[k], self.kernels[k,i], "full")
                
        de_dk_avg = np.mean(de_dk_store, axis=0)
        de_dx_avg = np.mean(de_dx_store, axis=0)

        self.kernels = self.optimizer.apply_optimizer(self.name+":K", self.kernels, de_dk_avg)
        self.biases = self.optimizer.apply_optimizer(self.name+":B", self.biases, de_db)
        return de_dx_avg

