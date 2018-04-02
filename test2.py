import os
import numpy as np
import tensorflow as tf

path = os.path.join('data', 'fashion-mnist')

fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainY = loaded[8:].reshape((60000)).astype(np.int32)

trX = trainX[:55000] / 255.
trY = trainY[:55000]

valX = trainX[55000:, ] / 255.
valY = trainY[55000:]


print(trX.size)
print(trainX)
