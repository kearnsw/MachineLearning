__author__ = 'will'

''' This algorithm handles the task of K-means clustering. This is achieved through a recursive process with 4
basic steps:
1) Initialize K random centroids.
2) Assign vectors to a cluster.
3) Move centroids.
4) Repeat steps 2 and 3 for a given number of iterations
'''

import matplotlib.pyplot as plt
import random
import math

random.seed()                   # seed rng to system time by default

class Clusters:

    def __init__(self, k, m):
        self.num = k
        self.index = [0] * m         # holds the cluster index for each training set
        self.centroids = []     # holds k centroids


def min_cost(example, centroids, k, dim):

    minimum = [100000, -1]

    for i in range(k):           # outer loop to determine the minimum cycles through all centroids
        total = 0

        for j in range(dim):                                # Calculate distance from training set to centroid
            total += (example[j] - centroids[i][j])**2      # loops through all dimensions
        distance = math.sqrt(total)                         # dist = sqrt((Ax-Bx)^2 + (Ay-By)^2 + ... (An-Bn)^2)
        if distance < minimum[0]:
            minimum[0] = distance       # stores value of the minimum distance
            minimum[1] = i              # stores index of the minimum distance

    return minimum[1]

def move_centroid(index_set, training_set, centroid_index, dim, m):

    total = [0] * dim
    new_centroid = [0] * dim
    count = 0
# Find mean of all points in a given cluster
    for i in range(m):                              # Loop through the entire training set
        if index_set[i] == centroid_index:
            for j in range(dim):
                total[j] += training_set[i][j]
            count += 1
    if count != 0:
        for i in range(dim):
            new_centroid[i] = total[i]/count

    return new_centroid

def plot_2d(training_set, clustering_index):

    for i in range(len(clustering_index)):
        if clustering_index[i] == 0:
            plt.scatter(training_set[i][0], training_set[i][1], color='yellow')
        if clustering_index[i] == 1:
            plt.scatter(training_set[i][0], training_set[i][1], color='blue')
        if clustering_index[i] == 2:
            plt.scatter(training_set[i][0], training_set[i][1], color='green')
        if clustering_index[i] == 3:
            plt.scatter(training_set[i][0], training_set[i][1], color='red')
    plt.show()

def kmeans(k, training_set, runs):
    m = len(training_set)
    dim = len(training_set[0])    # calculates the dimensionality of the training set
    clustering = Clusters(k, m)
    random_indices = []

    # Initialize random centroids at random training set values
    for i in range(k):
        for j in range(k*10):
            random_indices.append(random.randint(0, m-1))
            unique = list(set(random_indices))
        clustering.centroids.append(training_set[unique[random.randint(0, len(unique)-1)]])

    for i in range(runs):
        print('Initiating run ' + repr(i + 1) + ' of ' + repr(runs) + ' ...')
        for j in range(m):          # cluster assignment step
            clustering.index[j] = min_cost(training_set[j], clustering.centroids, k, dim)

        for j in range(k):          # move centroid step
            clustering.centroids[j] = move_centroid(clustering.index, training_set, j, dim, m)

    return clustering.index
