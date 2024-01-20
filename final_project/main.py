import os
import time
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
The main purpose of this script is to extract the PNG files from the matlab files 
to obtain the masks in a format that takes up less space. I then upload the PNGs
to Google Drive to actually do the segmentation training and testing.
'''


def obtain_mask(file_name):
    '''
    The way this function is currently written is to parse all of the labels in the
    matlab files from the SNOW dataset. It then saves the masks as a PNG file for
    further upload in Google Drive. It also returns the masks in an array, where
    each node within the array is a unique mask. I did this initially because
    I didn't realize that the images and masks would take up 14 GB of memory, my
    machine only has 16 GB of memory.
    '''

    mask_save = 'data/masks/'
    try:
        mat = scipy.io.loadmat(file_name)
        map = mat['inst_map']
        map = cv2.convertScaleAbs(map)
        _, binary_image = cv2.threshold(map, 1, 255, cv2.THRESH_BINARY)

        f = file_name.split('/')
        f = f[2].split('.')
        path = mask_save + f[0] + '.png'
        print(path)
        cv2.imwrite(path, binary_image)

        return binary_image
    except:
        raise Exception("Could not load given file")


def load_data(directory, img=False):
    '''
    This function loads images and masks into memory.
    '''

    files = sorted(os.listdir(directory))

    data = []
    for fn in files:
        path = os.path.join(directory, fn)

        if not fn.startswith('.'):
            print(f"Processing file: {path}")
            if img:
                data.append(cv2.imread(path))
            else:
                data.append(obtain_mask(path))

    return data


if __name__ == '__main__':
    # Data directories
    mask_dir1 = 'data/mask_0_5000/'
    mask_dir2 = 'data/mask_5k_10k/'
    mask_dir3 = 'data/mask_10k_15k/'
    mask_dir4 = 'data/mask_15k_20k/'

    image_dir1 = 'data/img_0_5000/'
    image_dir2 = 'data/img_5k_10k/'
    image_dir3 = 'data/img_10k_15k/'
    image_dir4 = 'data/img_15k_20k/'

    # Load in data
    start = time.time()
    # imgs = load_data(image_dir1, img=True)
    masks = load_data(mask_dir1, img=False)
    print("Elapsed time: " + str(time.time() - start) + " seconds")
    print(masks[0].shape)

    # imgs = np.array(imgs)
    masks = np.array(masks)
    print(masks.shape)

    # img = cv2.imread('data/Image/seed0000.png')
    '''plt.imshow(cv2.cvtColor(imgs[10], cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    plt.imshow(masks[10], cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    plt.show()'''
