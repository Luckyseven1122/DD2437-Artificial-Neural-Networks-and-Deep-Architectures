from abc import ABC, abstractmethod
from .Optimizer import *


class Centroids:
    def __init__(self, matrix):
        self._matrix = matrix
        self._optimizer = None
        self._N = matrix.shape[1]

    @abstractmethod
    def get_fi(self, X, sigma):
        pass

    def _transfer_function(self, x, sigma):
        return np.exp((-(x-self._matrix)**2)/(2*sigma**2))

    def _calculate_fi(self, X, sigma):
        fi = np.zeros((X.shape[0], self._N))
        for i in range(X.shape[0]):
            fi[i,:] = self._transfer_function(X[i,:], sigma)
        return fi

class Fixed(Centroids):
    def __init__(self, matrix):
        Centroids.__init__(self, matrix)

    def get_fi(self, X, sigma):
        return self._calculate_fi(X, sigma)
