import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace

BLUE = 0
GREEN = 1
RED = 2


def glomus_tumor(tumor):
    tumor_r = cv2.split(tumor.copy())[RED]
    tumor_r = cv2.bilateralFilter(tumor_r, 9, 50, 50)
    tumor_r = cv2.bitwise_not(tumor_r)
    cv2.imwrite('../images/results/tumor_preprocess.jpg', tumor_r)
    plt.imshow(tumor_r, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    sigma = 0.9

    # Apply the laplacian filter to the image
    laplacian = (sigma ** 2) * gaussian_laplace(tumor_r, sigma=sigma)
    plt.imshow(laplacian, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    binary = cv2.convertScaleAbs(laplacian)
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binary, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    tumor_new = cv2.bilateralFilter(tumor.copy(), 9, 50, 50)
    cv2.imwrite('../images/results/tumor_kmeans_preprocess.jpg', tumor_new)

    tumor_b, tumor_g, tumor_r = cv2.split(tumor_new)
    tumor_b, tumor_g, tumor_r = tumor_b.flatten(), tumor_g.flatten(), tumor_r.flatten()
    tumor_data = np.column_stack((tumor_b, tumor_g, tumor_r))
    tumor_data2 = np.column_stack((tumor_b, tumor_r))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(tumor_data)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the data points and cluster centers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tumor_data[:, 0], tumor_data[:, 1], tumor_data[:, 2], c=clusters, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', label='centers')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show()

    tumor_gray = cv2.cvtColor(tumor, cv2.COLOR_BGR2GRAY)
    tumor_x, tumor_y = tumor_gray.shape
    tumor_gray = tumor_gray.flatten()
    for i in range(len(tumor_gray)):
        if clusters[i] == 0:
            tumor_gray[i] = 0
        else:
            tumor_gray[i] = 255
    tumor_gray = tumor_gray.reshape(tumor_x, tumor_y)
    cv2.imwrite('../images/results/tumor_kmeans_RGB.jpg', tumor_gray)

    plt.imshow(tumor_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(tumor_data2)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the data points and cluster centers
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(tumor_data2[:, 0], tumor_data2[:, 1], c=clusters, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='centers')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show()

    tumor_gray = cv2.cvtColor(tumor, cv2.COLOR_BGR2GRAY)
    tumor_x, tumor_y = tumor_gray.shape
    tumor_gray = tumor_gray.flatten()
    for i in range(len(tumor_gray)):
        if clusters[i] == 0:
            tumor_gray[i] = 0
        else:
            tumor_gray[i] = 255
    tumor_gray = tumor_gray.reshape(tumor_x, tumor_y)

    cv2.imwrite('../images/results/tumor_kmeans_RB.jpg', tumor_gray)
    plt.imshow(tumor_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()


def breast_cancer(cancer):
    cancer_r = cv2.split(cancer)[RED]
    cancer_r = cv2.bilateralFilter(cancer_r, 9, 50, 50)
    cancer_r = cv2.bitwise_not(cancer_r)
    cv2.imwrite('../images/results/cancer_preprocess.jpg', cancer_r)
    plt.imshow(cancer_r, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    sigma = 1.5

    # Apply the laplacian filter to the image
    laplacian = (sigma ** 2) * gaussian_laplace(cancer_r, sigma=sigma)
    plt.imshow(laplacian, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    tumor_r = cv2.convertScaleAbs(laplacian)
    _, tumor_r = cv2.threshold(tumor_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(tumor_r, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    cancer_new = cv2.bilateralFilter(cancer.copy(), 9, 50, 50)
    cv2.imwrite('../images/results/cancer_kmeans_preprocess.jpg', cancer_new)

    cancer_b, cancer_g, cancer_r = cv2.split(cancer_new)
    cancer_b, cancer_g, cancer_r = cancer_b.flatten(), cancer_g.flatten(), cancer_r.flatten()
    cancer_data = np.column_stack((cancer_b, cancer_g, cancer_r))
    cancer_data2 = np.column_stack((cancer_g, cancer_r))

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(cancer_data)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the data points and cluster centers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cancer_data[:, 0], cancer_data[:, 1], cancer_data[:, 2], c=clusters, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', label='centers')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show()

    target_cluster = np.argmin(np.sum(centers, axis=1))
    cancer_gray = cv2.cvtColor(cancer, cv2.COLOR_BGR2GRAY)
    cancer_x, cancer_y = cancer_gray.shape
    cancer_gray = cancer_gray.flatten()
    for i in range(len(cancer_gray)):
        if clusters[i] == target_cluster:
            cancer_gray[i] = 0
        else:
            cancer_gray[i] = 255
    cancer_gray = cancer_gray.reshape(cancer_x, cancer_y)
    cv2.imwrite('../images/results/cancer_kmeans_RGB.jpg', cancer_gray)

    plt.imshow(cancer_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(cancer_data2)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the data points and cluster centers
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(cancer_data2[:, 0], cancer_data2[:, 1], c=clusters, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='centers')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show()

    target_cluster = np.argmin(np.sum(centers, axis=1))
    cancer_gray = cv2.cvtColor(cancer, cv2.COLOR_BGR2GRAY)
    cancer_x, cancer_y = cancer_gray.shape
    cancer_gray = cancer_gray.flatten()
    for i in range(len(cancer_gray)):
        if clusters[i] == target_cluster:
            cancer_gray[i] = 0
        else:
            cancer_gray[i] = 255
    cancer_gray = cancer_gray.reshape(cancer_x, cancer_y)
    cv2.imwrite('../images/results/cancer_kmeans_RG.jpg', cancer_gray)

    plt.imshow(cancer_gray, interpolation='nearest', cmap='gray')  # for grayscale images
    plt.show()


if __name__ == '__main__':
    img = '../images/GlomusTumor6.jpg'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    glomus_tumor(img)

    img = '../images/metastatic-breast-cancer.jpg'
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    breast_cancer(img)
