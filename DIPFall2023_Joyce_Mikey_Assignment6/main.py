import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


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
    img_fill = Fill(im, 500, 1, 1000, 800, 150)
    print(str(time.time() - start) + " s")

    cv2.imwrite('images/Naka1_fill.jpg', img_fill)
    #plt.imshow(cv2.cvtColor(img_fill, cv2.COLOR_BGR2RGB))
    #plt.show()


def kernel(x=5, sigma=0.1):
    axis = np.square(np.linspace(-(x-1)/2., (x-1)/2., x))
    gaussian = np.exp((-0.5*axis)/np.square(sigma))
    k = np.outer(gaussian, gaussian)

    return k/np.sum(k)


if __name__ == '__main__':
    img = 'images/Naka1_small.tif'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # partA(img)
    Ix = np.diff(img.copy(), axis=1)
    plt.imshow(cv2.cvtColor(Ix, cv2.COLOR_BGR2RGB))
    plt.show()

    Iy = np.diff(img.copy(), axis=0)
    plt.imshow(cv2.cvtColor(Iy, cv2.COLOR_BGR2RGB))
    plt.show()

    Ix, Iy = Ix[:-1, :], Iy[:, :-1]
    M = np.sqrt((np.square(Ix) + np.square(Iy)).astype(float)).astype(np.uint8)
    M_show = np.clip(10 * M, 0, 255).astype(np.uint8)
    plt.imshow(cv2.cvtColor(M_show, cv2.COLOR_BGR2RGB))
    plt.show()

    theta = np.degrees(np.arctan(Iy, Ix, casting='unsafe'))
    print('Angles: ', theta)

    plt.hist(theta.flatten(), bins=360, range=(0, 360))
    plt.show()

