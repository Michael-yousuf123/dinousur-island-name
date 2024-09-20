import numpy as np

def softmax(x):

    exp = np.exp(x - np.max(x))
    prob = exp/np.sum(exp)
    return prob