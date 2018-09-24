from abc import ABC, abstractmethod
import numpy as np

class Initializer:
    def __init__(self):
        pass

    @abstractmethod
    def new(self, shape):
        pass


class RandomNormal(Initializer):
    def new(self, shape):
        return np.random.normal(0, 0.1, shape)

class Zeros(Initializer):
    def new(shape):
        return np.zeros(shape)
