import os
import cv2
import numpy as np
import tensorflow as tf


a = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
              1, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
              3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
              3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
              4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6])
label_idx = []
#for i in range(10000*3073):
#    if i % (3072 + 1) == 0:
#        label_idx.append(i)

#print(label_idx)
labels = a[0::13]
print(labels.astype(np.int32))
a = np.delete(a, np.arange(0, 13*5, 13))
img = a.reshape((5, 3, 2, 2))

#t = tf.transpose(img, [0, 2, 3, 1])


fd = open(os.path.join('data', 'cifar-10-batches-bin/data_batch_1.bin'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
fd = open(os.path.join('data', 'cifar-10-batches-bin/data_batch_2.bin'))
next_loaded = np.fromfile(file=fd, dtype=np.uint8)
loaded = np.concatenate((loaded, next_loaded))


lables = loaded[0::3073]
loaded = np.delete(loaded, np.arange(0, 3073*20000, 3073))
trainX = loaded.reshape((20000, 3, 32, 32)).astype(np.float32)

trX = trainX[:17000] / 255
trY = lables[:17000]

valX = trainX[17000:] / 255
valY = lables[17000:]

print(lables.size)
print(trainX.size)

