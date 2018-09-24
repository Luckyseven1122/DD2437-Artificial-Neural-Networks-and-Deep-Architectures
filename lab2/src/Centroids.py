from abc import ABC, abstractmethod



class Centroids:
    def __init__(self, matrix):
        self._matrix = matrix

    @abstractmethod
    def calculate_fi(self):
        pass



class Fixed(Centroids):
    def __init__(self, matrix):
        Centroids.__init__(self, matrix)

    def calculate_fi(self):
        return self._matrix
