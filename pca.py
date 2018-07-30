import os
import glob
import numpy as np
import numpy.linalg as linalg
import cv2 as cv
from sklearn.decomposition import PCA, IncrementalPCA
from aug import helper


def calculate_pca(query, batch_size):
    paths = glob.glob(os.path.expanduser(query))
    eigen_vecs, eigen_vals, cov = get_ipca(paths, batch_size=batch_size)
    print("Eigen vectors:\n", eigen_vecs)
    print("Eigen vals:\n", eigen_vals)
    print("Covariance matrix\n", cov)


def get_numpy_pca(paths):
    """
    Computes PCA for BGR values of given images, using just numpy
    :param paths:
    :return: eigen_vectors, eigen_values, covariance matrix, order is BGR
    """
    all_pixels = None
    for i, path in enumerate(helper.show_progress(paths, 100)):

        # Load image and convert to a vector of RGB values
        pixels = load_image_pixels(path)

        # Accumulate pixels
        if all_pixels is None:
            all_pixels = pixels
        else:
            all_pixels = np.concatenate((all_pixels, pixels), axis=0)

    cov = np.cov(all_pixels.T)
    eigen_vals, eigen_vecs = linalg.eig(cov)

    return eigen_vecs, eigen_vals, cov


def get_pca(paths):
    """
    Computes PCA for BGR values of given images
    :param paths:
    :return: eigen_vectors, eigen_values, covariance matrix, order is BGR
    """
    pca = PCA(n_components=3)
    all_pixels = None
    for i, path in enumerate(helper.show_progress(paths, 100)):

        # Load image and convert to a vector of RGB values
        pixels = load_image_pixels(path)

        # Accumulate pixels
        if all_pixels is None:
            all_pixels = pixels
        else:
            all_pixels = np.concatenate((all_pixels, pixels), axis=0)

    pca.fit(all_pixels)
    cov = pca.get_covariance()
    eigen_vals, eigen_vecs = linalg.eig(cov)

    return eigen_vecs, eigen_vals, cov


def get_ipca(paths, batch_size):
    """
    Computes IPCA for BGR values of given images
    :param paths:
    :param batch_size:
    :return: eigen_vectors, eigen_values, covariance matrix, order is BGR
    """
    ipca = IncrementalPCA(n_components=3)
    all_pixels = None
    count = len(paths)
    for i, path in enumerate(helper.show_progress(paths, 100)):

        # Load image and convert to a vector of RGB values
        pixels = load_image_pixels(path)

        # Accumulate pixels
        if all_pixels is None:
            all_pixels = pixels
        else:
            all_pixels = np.concatenate((all_pixels, pixels), axis=0)

        # IPCA
        if i % batch_size-1 == 0 or i == count - 1:
            ipca.partial_fit(all_pixels)
            all_pixels = None

    cov = ipca.get_covariance()
    eigen_vals, eigen_vecs = linalg.eig(cov)

    return eigen_vecs, eigen_vals, cov


def load_image_pixels(path):
    img = cv.imread(path)
    pixels = img.reshape(-1, 3)
    pixels /= 255
    return pixels
