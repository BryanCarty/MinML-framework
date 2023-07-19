import numpy as np
class Flatten():
    def __init__(self, name):
        self.name = name
    def forward(self, input):
        self.input = input
        num_batches, num_channels, pixel_width, pixel_height = input.shape
        output = np.zeros(shape=(num_batches, num_channels*pixel_width*pixel_height))
        for b in range(num_batches):
            for c in range(num_channels):
                for w in range(pixel_width):
                    for h in range(pixel_height):
                        np.append(output[b], input[b][c][w][h])
        return output
    def backward(self, de_dy):
        return np.reshape(de_dy, newshape=(self.input.shape[1], self.input.shape[2], self.input.shape[3]))

