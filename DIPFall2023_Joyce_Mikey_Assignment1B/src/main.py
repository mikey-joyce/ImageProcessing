import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import time


def myimhist(bin_size, img, hist):
    # the hist variable that is passed will be the histogram
    if type(img) is not list:
        img = img.flatten().tolist()

    for i in range(len(img)):
        hist[int(img[i]/bin_size)] += 1

    return hist


def get_histogram(file_path, nbins, mask=None):
    '''
    This function will compute the histogram for an image based on the number of bins given.
    If a mask is given, then it will compute the histogram including the mask.
    If None is returned, then there was some sort of error with the inputs to the function.
    '''
    # ensure the number of bins is valid
    if nbins > 256 or nbins < 1:
        print('Max amount of bins is 256, minimum amount of bins is 1')
        return None

    # ensure the image exists
    if type(file_path) is str:
        img = read_img(file_path)
    else:
        img = file_path

    if mask is not None:
        mask = read_img(mask, 1)

    bin_size = (256 / nbins)

    if len(img.shape) > 2:
        # the image is not grayscale, most likely RGB
        if len(img.shape) > 3:
            print("Can't interpret image, please input an RGB or grayscale image")
            return None

        h_blue, h_green, h_red = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)

        img_b, img_g, img_r = cv2.split(img)
        temp_b, temp_g, temp_r = [], [], []

        if mask is not None:
            if mask.shape != img_b.shape:
                print('Mask is not same size as input image')
                return None
            else:
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i][j] > 0:
                            temp_b.append(img_b[i][j])
                            temp_g.append(img_g[i][j])
                            temp_r.append(img_r[i][j])
                img_b = temp_b
                img_g = temp_g
                img_r = temp_r

        h_blue = myimhist(bin_size, img_b, h_blue)
        h_green = myimhist(bin_size, img_g, h_green)
        h_red = myimhist(bin_size, img_r, h_red)

        h = [
            h_blue, h_green, h_red
        ]
    else:
        # the image is grayscale
        h = np.zeros(nbins)
        temp_img = []

        if mask is not None:
            if mask.shape != img.shape:
                print('Mask is not same size as input image')
                return None
            else:
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i][j] > 0:
                            temp_img.append(img[i][j])
                img = temp_img

        h = myimhist(bin_size, img, h)

    return h


def read_img(file_path, is_mask=None):
    if is_mask is None:
        # ensure the image exists
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            img.shape
        except:
            print('Input image did not exist\nProcessing a general image instead')
            img = cv2.imread('../images/jerry_gray.jpg', cv2.IMREAD_UNCHANGED)
    else:
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            img.shape
        except:
            print('Input mask did not exist\nProcessing a general mask instead')
            img = cv2.imread('../images/mask_stop2.jpg', cv2.IMREAD_UNCHANGED)

    return img


def perform_discard(img, amount):
    if amount > 1:
        print('The amount to discard must be a percentage (range 0 - 1)')
        return None

    temp_img = img.copy()
    temp_img = temp_img.flatten().tolist()
    temp_img.sort()

    num_to_discard = int(len(temp_img) * amount)

    last = temp_img[num_to_discard]
    while True:
        num_to_discard += 1
        if last == temp_img[num_to_discard]:
            last = temp_img[num_to_discard]
        else:
            lower_bound = temp_img[num_to_discard]
            break

    last = temp_img[-num_to_discard]
    while True:
        num_to_discard += 1
        if last == temp_img[-num_to_discard]:
            last = temp_img[-num_to_discard]
        else:
            upper_bound = temp_img[-num_to_discard]
            break

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < lower_bound:
                img[i][j] = lower_bound
            elif img[i][j] > upper_bound:
                img[i][j] = upper_bound


def min_max_stretch(img):
    minimum, maximum = np.min(img), np.max(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 255*(img[i][j] - minimum)/(maximum-minimum)


def print_intensities(img):
    print('Minimum: ', np.min(img))
    print('Maximum: ', np.max(img))


def plot_rgb_hist(h):
    plt.bar(range(len(h[0])), h[0], color='#3ec6f7')
    plt.show()
    plt.bar(range(len(h[1])), h[1], color='#3ef74d')
    plt.show()
    plt.bar(range(len(h[2])), h[2], color='#f74a3e')
    plt.show()


def histogram_experiment(file_path, nbins, mask=None):
    start_time = time.time()
    if mask is None:
        h = get_histogram(file_path, nbins)
        print("--- %s seconds ---" % (time.time() - start_time))
        if len(h) > 1:
            plot_rgb_hist(h)
        else:
            plt.bar(range(len(h)), h, color='#932bcf')
    else:
        h = get_histogram('../images/rgb_stop.jpg', nbins, '../images/mask_stop2.png')
        print("--- %s seconds ---" % (time.time() - start_time))
        plot_rgb_hist(h)


def contrast_experiment(file_path, write_name, percentage=None, original=None):
    image = read_img(file_path, cv2.IMREAD_UNCHANGED)

    if percentage is not None:
        perform_discard(image, percentage)
        print_intensities(image)

    if original is None:
        min_max_stretch(image)
        cv2.imwrite(write_name, image)

    if percentage is None:
        print_intensities(image)

    h = get_histogram(image, 256)
    plt.bar(range(len(h)), h, color='#932bcf')
    plt.show()
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()


if __name__ == '__main__':
    # Experiments for histogram calculation; uses helper function to mitigate written code
    num_bins = 256
    print('Entire Image Histogram nbins=256')
    histogram_experiment('../images/rgb_stop.jpg', num_bins)
    print('Regional Image Histogram nbins=256')
    histogram_experiment('../images/rgb_stop.jpg', num_bins, '../images/mask_stop2.png')

    num_bins = 16
    print('Entire Image Histogram nbins=16')
    histogram_experiment('../images/rgb_stop.jpg', num_bins)
    print('Regional Image Histogram nbins=16')
    histogram_experiment('../images/rgb_stop.jpg', num_bins, '../images/mask_stop2.png')

    # Experiments for contrast enhancement; uses helper function to mitigate written code
    print('Original Image')
    contrast_experiment('../images/night_gray.png', '', original=1)
    print('Min-Max Linear Stretch Experiment')
    contrast_experiment('../images/night_gray.png', '../images/results/Contrast Enhancement/night_gray_stretch.png')
    print('1% Discard Experiment')
    contrast_experiment('../images/night_gray.png', '../images/results/Contrast Enhancement/night_gray_p=0.01.png', 0.01)
    print('5% Discard Experiment')
    contrast_experiment('../images/night_gray.png', '../images/results/Contrast Enhancement/night_gray_p=0.05.png', 0.05)

    print('Done')
