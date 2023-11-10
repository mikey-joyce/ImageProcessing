import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math
from scipy.signal import convolve2d


def affine_helper(im, translation):
    new = np.zeros_like(im)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x, y, _ = np.matmul(translation, np.float32([[i], [j], [1]])).astype(int)

            if 0 <= x < im.shape[0] and 0 <= y < im.shape[1]:
                new[x[0], y[0]] = im[i, j]

    return new


def Translate(im, tx, ty):
    translation = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return affine_helper(im, translation)


def CropScale(im, x1, y1, x2, y2, s):
    im = im[x1:x2, y1:y2]
    translation = np.float32([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    # prove that it is cropped with the code below
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()
    return affine_helper(im, translation)


def scale(im, s):
    translation = np.float32([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    return affine_helper(im, translation)


def Vertical_Flip(im):
    return im[::-1, :]


def Horizontal_Flip(im):
    return im[:, ::-1]


def Rotate(im, angle):
    angle = angle * (np.pi/180)
    translation = np.float32([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    # print(translation)
    return affine_helper(im, translation)


def Fill(im, x1, y1, x2, y2, val):
    fill = im[x1:x2, y1:y2]
    for i in range(fill.shape[0]):
        for j in range(fill.shape[1]):
            fill[i, j] = val
    return im


def partA(im):

    start = time.time()
    img_translated = Translate(im, 500, 400)
    print(str(time.time() - start) + " s")
    cv2.imwrite('images/Naka1_translated.jpg', img_translated)
    #plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
    #plt.show()

    start = time.time()
    img_cropscale = CropScale(im, 500, 1, 1000, 800, 0.5)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_cropscale.jpg', img_cropscale)
    #plt.imshow(cv2.cvtColor(img_cropscale, cv2.COLOR_BGR2RGB))
    #plt.show()

    start = time.time()
    img_vflip = Vertical_Flip(im)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_vflip.jpg', img_vflip)
    #plt.imshow(cv2.cvtColor(img_vflip, cv2.COLOR_BGR2RGB))
    #plt.show()

    start = time.time()
    img_hflip = Horizontal_Flip(im)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_hflip.jpg', img_hflip)
    #plt.imshow(cv2.cvtColor(img_hflip, cv2.COLOR_BGR2RGB))
    #plt.show()

    start = time.time()
    img_rotated = Rotate(im, 30)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_rotated.jpg', img_rotated)
    #plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    #plt.show()

    start = time.time()
    img_fill = Fill(im.copy(), 500, 1, 1000, 800, 150)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_fill.jpg', img_fill)
    #plt.imshow(cv2.cvtColor(img_fill, cv2.COLOR_BGR2RGB))
    #plt.show()


def partB(im, scale=1):
    if scale == 1:
        name1 = 'images/Naka1_Ix.jpg'
        name2 = 'images/Naka1_Iy.jpg'
        name3 = 'images/Naka1_M.jpg'
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        name1 = 'images/Naka1_Ix_large.jpg'
        name2 = 'images/Naka1_Iy_large.jpg'
        name3 = 'images/Naka1_M_large.jpg'
        sobel_x = np.array([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
        sobel_y = np.array([[2, 2, 4, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -2, -1, -1], [-2, -2, -4, -2, -2]])

    Ix = convolve2d(im, sobel_x, mode='same', boundary='symm')
    cv2.imwrite(name1, Ix)
    #plt.imshow(cv2.cvtColor(Ix, cv2.COLOR_BGR2RGB))
    #plt.show()

    Iy = convolve2d(im, sobel_y, mode='same', boundary='symm')
    cv2.imwrite(name2, Iy)
    #plt.imshow(cv2.cvtColor(Iy, cv2.COLOR_BGR2RGB))
    #plt.show()

    M = np.sqrt((np.square(Ix) + np.square(Iy)))
    cv2.imwrite(name3, M)
    #plt.imshow(cv2.cvtColor(M_show, cv2.COLOR_BGR2RGB))
    #plt.show()

    theta = []
    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            angle = math.degrees(math.atan2(Ix[i][j], Iy[i][j]))
            if angle < 0:
                angle += 360
            theta.append(angle)

    plt.hist(theta, bins=360, range=(0, 360))
    plt.show()


if __name__ == '__main__':
    img = 'images/Naka1_small.tif'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    partA(img)
    partB(img)
    partB(img, 2)
