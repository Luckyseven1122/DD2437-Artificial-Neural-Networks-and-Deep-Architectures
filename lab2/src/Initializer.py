from abc import ABC, abstractmethod
import numpy as np

class Initializer:
    def __init__(self):
        pass

    @abstractmethod
    def new(self, shape):
        pass

class RandomNormal(Initializer):
    def __init__(self, std=0.1):
        self.__name__ = 'RandomNormal(std=' + str(std) + ')'
        self._std = std

    def new(self, shape):
        return np.random.normal(0, self._std, shape)

class Zeros(Initializer):
    def __init__(self):
        self.__name__ = 'Zeros'

    def new(shape):
        return np.zeros(shape)
