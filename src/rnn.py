import numpy as np
from activation import *
class RNN:
    '''
    '''
    def __init__(self, X, hidden_size):
        self.Wh = 0.01*np.random.randn(hidden_size, hidden_size)
        self.Wx = 0.01*np.random.randn(hidden_size, 1)
        self.Wy = 0.01*np.random.randn(hidden_size, hidden_size)
        self.bh = 0.01*np.random.randn(hidden_size, hidden_size)
        self.by = 0.01*np.random.randn(hidden_size, hidden_size)
        self.X = X
        self.T = max(X.shape)
        self.y = np.zeros(self.T, 1)
        self.hidden_size = hidden_size

    def forward(self, X, ht_1):
        self.X = X
        h, y, x = {}, {}, {}

        h[-1] = np.copy(ht_1)
        for t in range(len(self.X)):
            # getting the current input at the time
            x[t] = np.zeros((self.vocab_size, 1))
            if self.X != None:
                x[t][self.X[t]] = 1
            out = np.dot(x[t], self.Wx) + np.dot(h[t-1], self.Wh) + self.bh
            h[t] = np.tanh(out)
            y[t] = softmax(np.dot(h[t], self.Wy)) + self.by
        return h, y, x
    def backward(self):
        pass
    def sample():
        pass
    def loss(self, y_hat, y_target):
        """
        """
        return sum(-np.log(y_hat[t][y_target[t], 0] for t in range(len(self.X))))
    def update_parameter():
        pass 
    
    