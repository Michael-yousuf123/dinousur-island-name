import os
import numpy as np
from config import *
class DataCreation:
    """
    """
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            data = f.read().lower()
        self.chars = list(set(data))
        self.vocab_size = len(self.chars)
        self.ix_to_ch = {i:ch for i, ch in enumerate(self.chars)}
        self.ch_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        with open(path) as f:
            inputs = f.readline().strip().lower()
        self.inputs = [x for x in inputs]
    def generate_inputs(self, index):
        """
        """
        input_chars = self.inputs[index]
        inputs_char_idx = [self.ch_to_ix[char] for char in input_chars]
        # add as the last character in the output array
        X = [self.ch_to_ix['\n']] + inputs_char_idx
        Y = inputs_char_idx + [self.ch_to_ix['\n']]
        
        return np.array(X), np.array(Y)