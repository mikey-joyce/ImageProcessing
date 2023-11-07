import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def rgb2gray():
    if len(sys.argv) == 3:
        input_imageName = sys.argv[1]
        output_imageName = sys.argv[2]

        try:
            img = cv2.imread(input_imageName)
            img.shape
        except:
            print('Input image did not exist\nProcessing a general image instead')
            img = cv2.imread('../images/hela_cells_RGB.jpg')

        gray_shape = (img.shape[0], img.shape[1], 1)  # get grayscale shape
        gray = np.zeros(gray_shape)

        # formula: gray_xy = 0.114*blue_xy + 0.5871*green_xy + 0.2989*red_xy
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                blue, green, red = img[x][y][0], img[x][y][1], img[x][y][2]
                gray[x][y] = (0.114*blue) + (0.5871*green) + (0.2989*red)

        try:
            cv2.imwrite(output_imageName, gray)
        except:
            print('Output image file was not a valid file')
    else:
        print('Program not used correctly')
        print('The command prompt is: python3 rgb2gray input_file output_file')


if __name__ == '__main__':
    rgb2gray()
    print('Done')
