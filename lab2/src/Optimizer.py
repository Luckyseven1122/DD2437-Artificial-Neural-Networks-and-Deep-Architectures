from abc import ABC, abstractmethod
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self, fi, Y):
        pass

    @abstractmethod
    def loss(self, fi, W, Y):
        pass

    def output(self, fi, W):
        return np.dot(fi, W)

    def set_dims(self, N, M, O):
        self._N = N
        self._M = M
        self._O = O

class LeastSquares(Optimizer):
    def __init__(self):
        self.__name__ = 'LeastSquares'

    def train(self, fi, W, Y):
        return np.dot(np.linalg.pinv(fi), Y)

    def loss(self, fi, W, Y):
        return float(np.linalg.norm(np.dot(fi, W) - Y)**2)


class DeltaRule(Optimizer):
    def __init__(self, eta):
        self.eta = eta
        self.__name__ = 'DeltaRule(eta=' + str(eta) + ')'

    def train(self, fi, W, Y):
        fi = fi.reshape(-1, 1)
        delta_W = self.eta * (Y - np.dot(fi.T, W)) * fi
        return W + delta_W

    def loss(delf, fi, W, Y):
        '''
        Instantanious error for X_k
        '''
        fi = fi.reshape(-1, 1)
        Y = Y.reshape(1,-1)
        y = np.dot(fi.T, W).reshape(1,-1)
        return np.sum(((Y-y)**2)/2)


def test():
    pass

if __name__ == '__main__':
    test()
