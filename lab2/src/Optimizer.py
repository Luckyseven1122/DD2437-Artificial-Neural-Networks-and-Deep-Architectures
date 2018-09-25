from abc import ABC, abstractmethod
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self, fi, Y):
        pass

    @abstractmethod
    def loss(self, fi, w, Y):
        pass

class LeastSquares(Optimizer):
    def __init__(self):
        self.__name__ = 'LeastSquares'

    def train(self, fi, W, Y):
        y = np.dot(fi, W)
        return np.dot(np.linalg.pinv(fi), Y), y

    def loss(self, fi, w, Y):
        return np.linalg.norm(np.dot(fi, w) - Y)**2

class DeltaRule(Optimizer):
    def __init__(self, eta):
        self.eta = eta
        self.__name__ = 'DeltaRule'

    def train(self, fi, w, Y):

        pass

    def loss(delf, fi, w, Y):
        pass

def test():
    pass

if __name__ == '__main__':
    test()
