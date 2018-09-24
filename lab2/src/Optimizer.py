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
    def __name__(self):
        return 'LeastSquares'

    def __init__(self):
        pass

    def train(self, fi, W, Y):
        y = np.dot(fi, W)
        return np.dot(np.linalg.pinv(fi), Y), y

    def loss(self, fi, w, Y):
        return np.linalg.norm(np.dot(fi, w) - Y)**2

def test():
    pass

if __name__ == '__main__':
    test()
