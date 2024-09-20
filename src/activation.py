import numpy as np

def softmax(x):

    exp = np.exp(x - np.max(x))
    prob = exp/np.sum(exp)
    return prob
class Tanh:
    '''
    '''
    def forward(self, X):
        '''
        '''
        self.X = X
        self.outputs = np.tanh(X)
    def backward(self, dvalues):
        '''
        '''
        output = 1 - self.outputs**2
        self.dinputs = np.multiply(output, dvalues)
