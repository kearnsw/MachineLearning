__author__ = 'will'

''' This algorithm handles the task of Principal Component Analysis (PCA), i.e. the conversion of a
data set from N-dimensions to K-dimensions. This is accomplished in a 6-step process:
    1) Calculation of sample mean
    2) Subtraction of sample mean from each observation
    3) Calculation of covariance matrix, C
    4) Calculation of eigenvectors and eigenvalues of C
    5) Reduction to a subset of C, from N to K dimensions
    6) Projection of data
Input: a numpy array of dimension M x N
Output: a numpy array of dimension M x K
'''

import numpy as np
from scipy import linalg

def mean_adjust(array, m, n):

    mean_values = []

    # Calculate the mean of each dimension
    for i in xrange(n):
        total = 0
        for j in range(m):
            total += array[j][i]
        mean_values.append(total/m)

    # Subtract each dimensional mean from each vector
    for i in xrange(n):
        for j in range(m):
            array[j][i] -= mean_values[i]

    return array

def project_data(array, ei_rearranged, k):

    red_array = np.array([y[1] for y in ei_rearranged])
    projection = np.dot(array, red_array[:, :k])

    return projection


def pca(array, k):

    m = array.shape[0]
    n = array.shape[1]

    # Calculate the covariance of the mean adjusted dataset
    mean_adjusted = mean_adjust(array, m, n)
    co_matrix = np.cov(mean_adjusted.T)

    # Calculate eigenvectors and eigenvalues
    ei_values, ei_vectors = linalg.eig(co_matrix)

    ei_rearranged = sorted(zip(ei_values.real, ei_vectors.T), key=lambda x: x[0].real, reverse=True)

    return project_data(mean_adjusted, ei_rearranged, k)



