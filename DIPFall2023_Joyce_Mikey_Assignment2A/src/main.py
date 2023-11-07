import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def otsu(hist, img):
    p = np.divide(hist, np.sum(hist)).flatten()
    size = 255  # 255 instead of 256 because we don't want to compute the top border pixel
    mu = np.mean(img)

    sigmas = np.zeros(size)
    for t in range(1, size):  # we set 1 as start bc we don't want to compute bottom border pixel
        q1 = compute_q1(t, p)
        mu1 = compute_mu1(t, p)
        mu2 = compute_mu2(t, p, mu)

        sigmas[t] = q1 * (1 - q1) * ((mu1 - mu2)**2)

    return np.argmax(sigmas)


def compute_q1(t, p):
    if t == 1:
        return p[t]

    return compute_q1(t-1, p) + p[t]


def compute_mu1(t, p):
    if t == 1:
        return 0

    q1 = compute_q1(t, p)

    if q1 == 0:
        q1 = 0.0001

    return ((compute_q1(t-1, p) * compute_mu1(t-1, p)) + (t * p[t]))/q1


def compute_mu2(t, p, mu):
    q1 = compute_q1(t, p)

    return (mu - (q1 * compute_mu1(t, p)))/(1 - q1)


def binary_img(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255


def lung_experiment(file_path):
    # LUNG EXPERIMENT --> read image and histogram
    lung_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # show original image
    print('Lung experiment')
    plt.title('Original Image')
    plt.imshow(lung_img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    lung_hist = cv2.calcHist([lung_img], [0], None, [256], [0, 256]).flatten()

    # compute otsu and get the binary image
    start_time = time.time()
    thresh = otsu(lung_hist, lung_img)
    print("--- %s seconds ---" % (time.time() - start_time))

    binary_img(lung_img, thresh)

    # display the histogram with the threshold value
    print("Threshold: ", thresh)
    plt.title('Lung Histogram and Threshold value')
    plt.axvline(x=thresh, color='#2eb51f')
    plt.bar(range(len(lung_hist)), lung_hist, color='#932bcf')
    plt.show()

    # display the new image
    plt.title('New Image')
    plt.imshow(lung_img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()


def river_experiment(file_path, channel=0):
    # RIVER EXPERIMENT --> show original image first
    river_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    print('River experiment')
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(river_img, cv2.COLOR_BGR2RGB))  # for rgb images
    plt.show()

    river_img_channel = cv2.split(river_img)[channel]
    river_img_channel = cv2.bilateralFilter(river_img_channel, 9, 75, 75)  # pre processing

    river_hist = cv2.calcHist([river_img_channel], [0], None, [256], [0, 256]).flatten()

    # compute otsu and get the binary image
    start_time = time.time()
    thresh = otsu(river_hist, river_img_channel)
    print("--- %s seconds ---" % (time.time() - start_time))

    binary_img(river_img_channel, thresh)

    color = ''
    thresh_color = ''
    if channel == 2:
        color = '#eb3434'
        thresh_color = '#3ec6f7'
    elif channel == 1:
        color = '#3ef74d'
        thresh_color = '#8532a8'
    elif channel == 0:
        color = '#3ec6f7'
        thresh_color = '#eb3434'

    # display the histogram with the threshold value
    print("Threshold: ", thresh)
    plt.title('River Color Channel Histogram and Threshold value')
    plt.axvline(x=thresh, color=thresh_color)
    plt.bar(range(len(river_hist)), river_hist, color=color)
    plt.show()

    plt.title('New Image')
    plt.imshow(river_img_channel, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()


def square_experiment(file_path):
    # Square experiment
    square_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    plt.title('Original Image')
    plt.imshow(square_img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    square_hist = cv2.calcHist([square_img], [0], None, [256], [0, 256]).flatten()

    # compute otsu and get the binary image
    start_time = time.time()
    thresh = otsu(square_hist, square_img)
    print("--- %s seconds ---" % (time.time() - start_time))

    binary_img(square_img, thresh)

    # display the histogram with the threshold value
    print("Threshold: ", thresh)
    plt.title('Square Histogram and Threshold value')
    plt.axvline(x=thresh)
    plt.bar(range(len(square_hist)), square_hist, color='#eb3434')
    plt.show()

    plt.title('New Image')
    plt.imshow(square_img, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()


if __name__ == '__main__':
    # GET FILE PATHS
    river_path = '../images/river.jpg'
    lung_path = '../images/lungs.jpg'
    square_path = '../images/square.jpg'

    # RIVER EXPERIMENT
    river_experiment(river_path, 0)  # blue
    river_experiment(river_path, 1)  # green
    river_experiment(river_path, 2)  # red

    # LUNG EXPERIMENT
    lung_experiment(lung_path)

    # SQUARE EXPERIMENT
    square_experiment(square_path)
