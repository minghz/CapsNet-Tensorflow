import os
import scipy
import numpy as np
import tensorflow as tf


def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
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

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
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

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_cifar10(batch_size, is_training=True):
    path = os.path.join('data', 'cifar-10-batches-bin')
    if is_training:
        fd = open(os.path.join(path, 'data_batch_1.bin'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        fd = open(os.path.join(path, 'data_batch_2.bin'))
        next_loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = np.concatenate((loaded, next_loaded))
       # fd = open(os.path.join(path, 'data_batch_3.bin'))
       # next_loaded = np.fromfile(file=fd, dtype=np.uint8)
       # loaded = np.concatenate((loaded, next_loaded))
       # fd = open(os.path.join(path, 'data_batch_4.bin'))
       # next_loaded = np.fromfile(file=fd, dtype=np.uint8)
       # loaded = np.concatenate((loaded, next_loaded))
       # fd = open(os.path.join(path, 'data_batch_5.bin'))
       # next_loaded = np.fromfile(file=fd, dtype=np.uint8)
       # loaded = np.concatenate((loaded, next_loaded))

        lables = loaded[0::3073]
        loaded = np.delete(loaded, np.arange(0, 3073*20000, 3073))
        trainX = loaded.reshape((20000, 3, 32, 32)).astype(np.float32)

        trX = trainX[:17000] / 255
        trX = tf.transpose(trX, [0, 2, 3, 1])
        trX = tf.image.resize_image_with_crop_or_pad(trX, 28, 28)
        trX.set_shape([17000, 28, 28, 3])
        trX = tf.image.rgb_to_grayscale(trX)
        assert trX.get_shape() == [17000, 28, 28, 1]

        trY = lables[:17000]

        valX = trainX[17000:] / 255
        valX = tf.transpose(valX, [0, 2, 3, 1])
        valX = tf.image.resize_image_with_crop_or_pad(valX, 28, 28)
        valX.set_shape([3000, 28, 28, 3])
        valX = tf.image.rgb_to_grayscale(valX)
        assert valX.get_shape() == [3000, 28, 28, 1]

        valY = lables[17000:]

        num_tr_batch = 17000 // batch_size
        num_val_batch = 3000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'test_batch.bin'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        lables = loaded[0::3073]
        loaded = np.delete(loaded, np.arange(0, 3073*50000, 3073))

        teX = loaded.reshape((50000, 3, 32, 32)).astype(np.float)
        teY = lables.astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'cifar10':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_cifar10(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
