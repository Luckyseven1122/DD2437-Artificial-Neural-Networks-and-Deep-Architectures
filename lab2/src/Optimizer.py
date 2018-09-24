from abc import ABC, abstractmethod
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass



class LeastSquares(Optimizer):
    def __init__(self):
        pass

    def train(self, fi, Y):
        return np.dot(np.linalg.inv(np.dot(fi.T, fi.T)), np.dot(fi.T, Y))



def test():
    pass

if __name__ == '__main__':
    test()
