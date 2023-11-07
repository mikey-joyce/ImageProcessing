import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import time


def adaptivemedian(img, smax):
    filtered = np.copy(img)
    reconstruction = np.zeros((smax-1)//2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered[i][j], temp_index = get_new_pixel(img, smax, i, j)

            if temp_index > -1:
                reconstruction[temp_index] += 1

    return filtered, reconstruction


def get_new_pixel(img, smax, i, j):
    size = 3
    while size <= smax:
        shalf = size//2
        sxy = img[i - shalf:i + shalf + 1, j - shalf:j + shalf + 1]  # window
        sorted_sxy = np.sort(sxy.flatten())
        zxy = img[i][j]  # center pixel

        if len(sorted_sxy) < (size*size):
            return zxy, -1

        # local median, local min, local max
        zmed, zmin, zmax = sorted_sxy[len(sorted_sxy)//2], sorted_sxy[0], sorted_sxy[-1]

        if zmin < zmed < zmax:
            if zmin < zxy < zmax:
                return zxy, -1
            else:
                return zmed, ((size-1)//2)-1
        else:
            size += 2

    return zmed, ((size-3)//2)-1


def mse(img1, img2):
    if img1.shape != img2.shape:
        print('Images not same shape')
        return None

    return np.sum((img1 - img2) ** 2)/(img1.shape[0] * img1.shape[1])


def mse_filter_experiment(img1, img2, which_experiment=0):
    img = img2.copy()
    if which_experiment == 1:
        # do the gaussian filter with sigma = 2
        img = gaussian_filter(img, 2)
    elif which_experiment == 2:
        # do the gaussian filter with sigma = 7
        img = gaussian_filter(img, 7)
    elif which_experiment == 3:
        # do the median filter 7x7
        img = median_filter(img, (7, 7))
    elif which_experiment == 4:
        # do the median filter 19x19
        img = median_filter(img, (19, 19))

    print('MSE of the images: ', mse(img1, img))
    plt.imshow(img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    return img


def adaptive_median_experiment(img, smax, original):
    if smax == 19:
        x = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    elif smax == 7:
        x = [3, 5, 7]

    plt.imshow(img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    start_time = time.time()
    img, y = adaptivemedian(img, smax)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('MSE of the images: ', mse(original, img))

    plt.title('Reconstruction over time')
    plt.plot(x, y)
    plt.show()

    plt.imshow(img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    return img


if __name__ == '__main__':
    test1 = '../images/Test1.png'
    test1noise1 = '../images/Test1Noise1.png'
    test1noise2 = '../images/Test1Noise2.png'

    test2 = '../images/Test2.png'
    test2noise2 = '../images/Test2Noise2.png'

    test1 = cv2.imread(test1, cv2.IMREAD_UNCHANGED)
    test1noise1 = cv2.imread(test1noise1, cv2.IMREAD_UNCHANGED)
    test1noise2 = cv2.imread(test1noise2, cv2.IMREAD_UNCHANGED)

    test2 = cv2.imread(test2, cv2.IMREAD_UNCHANGED)
    test2noise2 = cv2.imread(test2noise2, cv2.IMREAD_UNCHANGED)

    # test1noise1 experiments
    print('test1noise1 experiments')
    print('Noisy Image')
    mse_filter_experiment(test1, test1noise1)

    print('Gaussian sigma=2')
    new_img = mse_filter_experiment(test1, test1noise1, 1)
    #cv2.imwrite('../images/results/test1noise1_gauss_s=2.jpg', new_img)

    print('Gaussian sigma=7')
    new_img = mse_filter_experiment(test1, test1noise1, 2)
    #cv2.imwrite('../images/results/test1noise1_gauss_s=7.jpg', new_img)

    print('Median 7x7')
    new_img = mse_filter_experiment(test1, test1noise1, 3)
    #cv2.imwrite('../images/results/test1noise1_median_s=7.jpg', new_img)

    print('Median 19x19')
    new_img = mse_filter_experiment(test1, test1noise1, 4)
    #cv2.imwrite('../images/results/test1noise1_median_s=19.jpg', new_img)
    print()

    # test1noise2 experiments
    print('test1noise2 experiments')
    print('Noisy Image')
    mse_filter_experiment(test1, test1noise2)

    print('Gaussian sigma=2')
    new_img = mse_filter_experiment(test1, test1noise2, 1)
    #cv2.imwrite('../images/results/test1noise2_gauss_s=2.jpg', new_img)

    print('Gaussian sigma=7')
    new_img = mse_filter_experiment(test1, test1noise2, 2)
    #cv2.imwrite('../images/results/test1noise2_gauss_s=7.jpg', new_img)

    print('Median 7x7')
    new_img = mse_filter_experiment(test1, test1noise2, 3)
    #cv2.imwrite('../images/results/test1noise2_median_s=7.jpg', new_img)

    print('Median 19x19')
    new_img = mse_filter_experiment(test1, test1noise2, 4)
    #cv2.imwrite('../images/results/test1noise2_median_s=19.jpg', new_img)
    print()

    # ADAPTIVE MEDIAN EXPERIMENTS
    print('adaptive median experiments')

    print('Test2 Noisy Image')
    mse_filter_experiment(test2, test2noise2)

    print('Adaptive Median Test1Noise2 7x7')
    #temp = test1noise2.copy()
    temp = adaptive_median_experiment(test1noise2, 7, test1)
    cv2.imwrite('../images/results/test1noise2_adaptivemedian_s=7.jpg', temp)

    print('Adaptive Median Test1Noise2 19x19')
    #temp = test1noise2.copy()
    temp = adaptive_median_experiment(test1noise2, 19, test1)
    cv2.imwrite('../images/results/test1noise2_adaptivemedian_s=19.jpg', temp)

    print('Adaptive Median TestNoise2 7x7')
    #temp = test2noise2.copy()
    temp = adaptive_median_experiment(test2noise2, 7, test2)
    cv2.imwrite('../images/results/test2noise2_adaptivemedian_s=7.jpg', temp)

    print('Adaptive Median Test2Noise2 19x19')
    #temp = test2noise2.copy()
    temp = adaptive_median_experiment(test2noise2, 19, test2)
    cv2.imwrite('../images/results/test2noise2_adaptivemedian_s=19.jpg', temp)
