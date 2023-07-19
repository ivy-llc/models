import numpy as np


def np_softmax(inputs):
    """Apply the softmax on the output"""
    return np.exp(inputs) / np.sum(np.exp(inputs))
