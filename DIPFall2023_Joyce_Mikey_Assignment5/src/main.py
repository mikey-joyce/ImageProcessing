import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statistics import mode

if __name__ == '__main__':
    img = '../images/GlomusTumor6_crp2.jpg'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    img = cv2.bilateralFilter(img, 9, 50, 50)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    img_b, img_g, img_r = cv2.split(img)
    img_b, img_g, img_r = img_b.flatten(), img_g.flatten(), img_r.flatten()
    data = np.column_stack((img_b, img_g, img_r))

    while True:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)
        clusters = kmeans.labels_
        centers = kmeans.cluster_centers_

        if mode(clusters) == 0:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', label='centers')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_x, img_y = img_gray.shape
    img_gray = img_gray.flatten()
    for i in range(len(img_gray)):
        if clusters[i] == 0:
            img_gray[i] = 0
        else:
            img_gray[i] = 255
    img_gray = img_gray.reshape(img_x, img_y)

    plt.imshow(img_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.title('K-means Foreground Mask')
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    img_gray = cv2.erode(img_gray, kernel, iterations=1)
    img_gray = cv2.dilate(img_gray, kernel, iterations=1)
    '''plt.imshow(img_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()'''

    dist = cv2.distanceTransform(img_gray, cv2.DIST_L2, 5)

    kernel = np.ones((3, 3), np.uint8)
    bg = cv2.dilate(img_gray, kernel, iterations=3)
    '''plt.imshow(bg, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()'''

    ret, fg = cv2.threshold(dist, 0.28 * dist.max(), 255, cv2.THRESH_BINARY)
    fg = cv2.erode(fg, kernel, iterations=1)
    fg = cv2.dilate(fg, kernel, iterations=1)
    fg = fg.astype(np.uint8)

    '''plt.imshow(fg, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()'''

    other = cv2.subtract(bg, fg)

    temp = np.zeros_like(fg)
    fg_color = np.stack((temp, fg, temp), axis=-1)
    blend = cv2.addWeighted(img, 0.7, fg_color, 0.5, 0)
    plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    plt.title('Cells markers overlayed on image')
    plt.show()

    fg = cv2.merge([fg.astype(np.int32)])
    fg += 1
    fg[other == 255] = 0

    boundary = cv2.watershed(img, fg)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(boundary)
    ax.axis('off')
    plt.title('Watershed Boundaries')
    plt.show()

    _, b = cv2.threshold(boundary.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, markers = cv2.connectedComponents(b)

    areas = []
    for i in range(1, ret):
        a = np.sum(markers == i)
        areas.append(a)

    markers += 1
    markers[other == 255] = 0
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('Connected Component Segmentation')
    ax.imshow(markers)
    ax.axis('off')
    plt.show()

    areas.pop(0)
    plt.hist(areas, bins=50, color='b', alpha=0.7)
    plt.title('Distribution of cell areas')
    plt.ylabel('Num Cells')
    plt.xlabel('Area in pixels')
    plt.show()


