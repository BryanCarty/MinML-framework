from MinML.Reader.reader import Reader
import pandas as pd
from MinML.Layers.ConvLayer import Conv
from MinML.Optimizers.GD_momentum import GD_Momentum
import numpy as np
from MinML.Activations.ReLU import ReLU
from MinML.Layers.MaxPooling import ValidMaxPooling
from MinML.Layers.Flatten import Flatten
from MinML.Layers.DenseLayer import Dense
from MinML.Activations.SoftMax import Softmax
from MinML.Losses.CrossEntropyLoss import CategoricalCrossEntropy
import sys



# Create data reader
d = pd.read_csv('archive/labels.csv')
reader = Reader(d, 128, 10, normalize_pixels=True, one_hot_encoding=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

GD_M = GD_Momentum(0.01, 0.9)

LOSS = CategoricalCrossEntropy("L1")

network = [
    Conv(input_shape=(128,96,96,3), kernel_shape=(4,4), num_kernels=40, stride=2, optimizer=GD_M, conv_mode="valid", kernel_initializer="he_normal", name="C1", input_layer=True),
    ReLU(name="R1"),
    ValidMaxPooling(3,2, name="MP1"),
    Conv(input_shape=(128,23,23,40), kernel_shape=(2,2), num_kernels=105, stride=1, optimizer=GD_M, conv_mode="valid", kernel_initializer="he_normal", name="C2"),
    ReLU(name="R2"),
    ValidMaxPooling(4,2, name="MP2"),
    Conv(input_shape=(128,10,10,105), kernel_shape=(2,2), num_kernels=158, stride=1, optimizer=GD_M, conv_mode="valid", kernel_initializer="he_normal", name="C3"),
    ReLU(name="R3"),
    Conv(input_shape=(128,9,9,158), kernel_shape=(2,2), num_kernels=158, stride=1, optimizer=GD_M, conv_mode="valid", kernel_initializer="he_normal", name="C4"),
    ReLU(name="R4"),
    Conv(input_shape=(128,8,8,158), kernel_shape=(2,2), num_kernels=105, stride=1, optimizer=GD_M, conv_mode="valid", kernel_initializer="he_normal", name="C5"),
    ReLU(name="R5"),
    ValidMaxPooling(3,2, name="MP3"),
    Flatten(name="F1"),
    Dense(input_neurons=945, output_neurons=945, optimizer=GD_M, name="D1"),
    ReLU(name="R6"),
    Dense(input_neurons=945, output_neurons=475, optimizer=GD_M, name="D2"),
    ReLU(name="R7"),
    Dense(input_neurons=475, output_neurons=8, optimizer=GD_M, name="D3"),
    Softmax(name="S1"),
]

loss_record = []

count = 0
while True:
    r = reader.next()
    if r is False:
        break

    x,y = r
    
    x = np.transpose(x, (0, 3, 1, 2))
    y = np.squeeze(y, axis=1)


    for layer in network:
        x = layer.forward(x)

    loss = LOSS.cross_entropy_loss(x,y)
    print('Loss: '+str(loss)+'\n')
    loss_record.append(loss)
    x = LOSS.cross_entropy_loss_prime()

    for layer in network[::-1]:
        x = layer.backward(x)

    if count >=2:
        break
    count+=1


# Open the file in write mode
with open('results/result.txt', 'w') as file:
    # Iterate over the list elements and write them to the file
    for item in loss_record:
        file.write(str(item) + '\n')  # Convert each element to string and add a new line

