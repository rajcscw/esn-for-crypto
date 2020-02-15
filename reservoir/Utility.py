import numpy as np


def formFeatureVectorsWithBias(data):
    n, d = data.shape
    input = data[:n-1]
    input = np.hstack((np.ones((n-1, 1)), input)).reshape((n-1, d+1))
    output = data[1:n]
    return input, output
