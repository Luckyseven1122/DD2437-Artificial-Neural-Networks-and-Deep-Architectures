import numpy as np
from .Initializer import Initializer
from .Centroids import Fixed

'''
    Input Layer : Size of the input layer determined by the dimensionality of the input.

            Input shape: n x M
                - n : # samples
                - M : # input units

    Hidden layer:

            Weight Matrix: N x 1
                - N : # hidden units

    Output layer:

            Output layer: n x 1


    Mapping: R^M -> R^N -> R^1

'''




class Network:
    def __init__(self, X, Y, hidden_nodes, sigma=1.0, centroids=None, initializer=None):
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

        # extract dimensionality information
        self.n_samples = X.shape[0]
        self.M_input_nodes = X.shape[1]
        self.N_hidden_nodes = hidden_nodes

        # Ensure N < n
        assert self.N_hidden_nodes < self.n_samples

        self.linear_weights = initializer.new((self.N_hidden_nodes, 1))


    def _transfer_function(self, x, mu, sigma):
        return np.exp((-(x-mu)**2)/(2*sigma**2))

    def _calculate_fi(self):
        for row in range(self.N_hidden_nodes):


    def train(self, epochs, optimizer=None):
        assert optimizer != None


        self.linear_weights = centroids.calculate_fi()

        for e in range(epochs):


        # initialize radius centroids weights




def test():

    pass


if __name__ == '__main__':
    test()
