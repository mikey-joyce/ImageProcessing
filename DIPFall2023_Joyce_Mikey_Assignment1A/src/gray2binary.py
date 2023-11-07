import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def gray2binary():
    if len(sys.argv) == 4:
        input_imageName = sys.argv[1]
        output_imageName = sys.argv[2]
        threshold = int(sys.argv[3])

        if 0 <= threshold <= 255:
            try:
                img = cv2.imread(input_imageName, cv2.IMREAD_UNCHANGED)
                img.shape
            except:
                print('Input image did not exist\nProcessing a general image instead')
                img = cv2.imread('../images/hela_cells_gray.jpg', cv2.IMREAD_UNCHANGED)

            if len(img.shape) == 2:
                # manipulate values based on threshold to either be 0 or 255
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        if img[x][y] < threshold:
                            img[x][y] = 0
                        else:
                            img[x][y] = 255

                try:
                    cv2.imwrite(output_imageName, img)
                except:
                    print('Output image file was not a valid file')
            else:
                print('Image provided was not grayscale')
        else:
            print('Threshold not in range 0-255')
    else:
        print('Program not used correctly')
        print('The command prompt is: python3 gray2binary input_file output_file threshold_value')


if __name__ == '__main__':
    gray2binary()
    print('Done')
