import os
import numpy as np
from matplotlib import pyplot as plt

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def process_data(filename, dest_dir):
    fd = open(filename)
    loaded = np.fromfile(file=fd, dtype=np.uint8)

    lables = loaded[0::3073]
    loaded = np.delete(loaded, np.arange(0, 3073*10000, 3073))
    loaded = loaded.reshape((10000, 3072)).astype(np.float32)

    #create or append to label file
    label_file = open(os.path.join(dest_dir, 'labels.bin'), "a+b")
    label_file.write(bytes(lables.tolist()))
    label_file.close()

    #create or append to image file
    #greyscale
    gray_images = np.array([])
    for img in loaded:
        scaled_red = 0.2989 * img[0:1024]
        scaled_green = 0.5870 * img[1024:2048]
        scaled_blue = 0.1140 * img[2048:3072]
        gray = np.array([scaled_red, scaled_green, scaled_blue]).sum(axis=0)
        
        gray_images = np.append(gray_images, gray, axis=0)

    gray_images = gray_images.reshape((10000, 32, 32)).astype(np.float32)

    #crop
    cropped_images = np.array([])
    for img in gray_images:
        img = crop_center(img, 28, 28)
        img = img.reshape((784)).astype(int)
        #plt.imshow(img, cmap='gray')
        #plt.show()
        cropped_images = np.append(cropped_images, img, axis=0)

    cropped_images = cropped_images.reshape(10000*784).astype(int)

    image_file = open(os.path.join(dest_dir, 'images.bin'), "a+b")
    image_file.write(bytes(cropped_images.tolist()))
    image_file.close()


def main():
    source_dir = './cifar-10-batches-bin'
    # TODO: create the dir if it doesn't exist
    # os.path.isfile(source_dir)

    dest_dir = 'cifar-10-preprocessed'
    training_dir = dest_dir + '/train'
    test_dir = dest_dir + '/test'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    print('Processing 50,000 Training images')
    for i in range(1, 6):
        filename = source_dir + '/data_batch_' + str(i) + '.bin'
        print('Processing binary from: '+ filename)
        process_data(filename, training_dir)

    print('Processing 10,000 Testing images')
    filename = source_dir + '/test_batch.bin'
    print('Processing binary from: '+ filename)
    process_data(filename, test_dir)

    print('Done')


if __name__ == "__main__":
    main()
