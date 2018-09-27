from abc import ABC, abstractmethod
from .Optimizer import *
import matplotlib.pyplot as plt

class Centroids:
    def __init__(self, matrix, update):
        self._matrix = matrix
        self._N = matrix.shape[1]

    @abstractmethod
    def get_fi(self, X, sigma):
        pass

    @abstractmethod
    def update(self, X, sigma):
        pass

    def get_matrix(self):
        return self._matrix

    def _transfer_function(self, x, sigma):
        return np.exp((-(x-self._matrix)**2)/(2*sigma**2))

    def _calculate_fi(self, X, sigma):
        fi = np.zeros((X.shape[0], self._N))
        for i in range(X.shape[0]):
            fi[i,:] = self._transfer_function(X[i,:], sigma)
        return fi

    def _normalize_matrix(self):
        for c in range(self._matrix.shape[1]):
            self._matrix[:,c] = self._matrix[:,c] / np.linalg.norm(self._matrix[:,c])

    def _find_closest_RBF_unit_index(self, x):
        distances = np.zeros((1, self._matrix.shape[1]))
        for c in range(self._matrix.shape[1]):
            distances[:,c] = np.linalg.norm(self._matrix[:,c] - x)
        idx = np.argmin(distances)
        return distances, idx


class Fixed(Centroids):
    def __init__(self, matrix):
        super().__init__(matrix, update=False)
        self.__name__ = 'Fixed'

    def get_fi(self, X, sigma):
        return self._calculate_fi(X, sigma)


class VanillaCL(Centroids):
    def __init__(self, matrix, space=[-2, 2], eta=0.1):
        matrix = np.random.uniform(space[0], space[1], size=(matrix.shape))
        super().__init__(matrix, update=True)
        #self._c = 0
        #self._normalize_matrix()
        self.__name__ = 'VanillaCL'
        self._eta = eta

    def get_fi(self, X, sigma):
        sample = X[np.random.randint(0, X.shape[0]),:].reshape(-1,1) # MIGHT FUCK THINGS UP! .shape(-1,1)??
        _, idx = self._find_closest_RBF_unit_index(sample)
        self._matrix[:,idx,None] += self._eta*(sample - self._matrix[:,idx].reshape(-1,1))
        #plt.clf()
        #plt.axis([0, 6, -1, 1])
        #plt.scatter(self._matrix, np.zeros(self._matrix.shape[1]))
        #plt.savefig('./figures/task3.3/' + str(self._c) + '.png')
        #self._c += 1
        #self._normalize_matrix()
        return self._calculate_fi(X, sigma)



class LeakyCL(Centroids):
    def __init__(self, matrix, space=[-2, 2], eta=0.1):
        matrix = np.random.uniform(space[0], space[1], size=(matrix.shape))
        super().__init__(matrix, update=True)
        #self._c = 0
        #self._normalize_matrix()
        self.__name__ = 'VanillaCL'
        self._eta = eta

    def get_fi(self, X, sigma):
        sample = X[np.random.randint(0, X.shape[0]),:].reshape(-1,1) # MIGHT FUCK THINGS UP! .shape(-1,1)??
        distances, idx = self._find_closest_RBF_unit_index(sample)

        # Evaluate if this is good approach or not
        reverse_normalized_distances = 1 - distances / np.linalg.norm(distances)
        reverse_normalized_distances[:,idx] = np.sum(reverse_normalized_distances)
        reverse_normalized_distances_softmax = np.exp(reverse_normalized_distances) / np.sum(np.exp(reverse_normalized_distances), axis=1)
        self._matrix += self._eta * reverse_normalized_distances_softmax * (sample - self._matrix)
        return self._calculate_fi(X, sigma)
