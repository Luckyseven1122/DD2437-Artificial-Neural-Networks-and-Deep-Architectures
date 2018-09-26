from abc import ABC, abstractmethod
import numpy as np

class Initializer:
    def __init__(self):
        pass

    @abstractmethod
    def new(self, shape):
        pass

    def _storage(self, file, arr):
        try:
            return np.load(file)
        except IOError:
            np.save('./src/wcache/' + file, arr)
            return arr

class RandomNormal(Initializer):
    def __init__(self, std=0.1):
        self.__name__ = 'RandomNormal(std=' + str(std) + ')'
        self._std = std

    def new(self, shape):
        W = np.random.normal(0, self._std, shape)
        W = self._storage(self.__name__ + '_shape=' +str(shape), W)
        print(W.shape)
        return W


class Zeros(Initializer):
    def __init__(self):
        self.__name__ = 'Zeros'

    def new(shape):
        W = np.zeros(shape)
        W = self._storage(self.__name__ + '_shape=' + str(shape), W)
        return W
