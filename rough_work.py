import numpy as np
from scipy.signal import correlate2d, convolve2d

matrix_2 = np.zeros(shape=(96,96))
# Input matrix
matrix = np.ones(shape=(47,47))
print("Original matrix:")
print(matrix)

# Dilated matrix
dilated_matrix = np.zeros(((2 * matrix.shape[0])-1, (2 * matrix.shape[1]-1)))
dilated_matrix[::2, ::2] = matrix
print("\nDilated matrix:")
print(dilated_matrix.shape)



d =correlate2d(matrix_2, dilated_matrix, "valid")
print(d.shape)