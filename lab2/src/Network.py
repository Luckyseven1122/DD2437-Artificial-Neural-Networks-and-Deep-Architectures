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

        self.sigma = sigma

        # extract dimensionality information
        self.n_samples = X.shape[0]
        self.M_input_nodes = X.shape[1]
        self.N_hidden_nodes = hidden_nodes

        # Ensure N < n
        assert self.N_hidden_nodes < self.n_samples

        self.fi = None
        self.centroids = centroids
        self.linear_weights = initializer.new((self.N_hidden_nodes, 1))


        # Metrics placement
        self.training_loss = []
        self.validation_loss = []

    def train(self, epochs, optimizer=None):
        self.optimizer = optimizer
        self.fi = self.centroids.get_fi(self.X, self.sigma)

        for e in range(epochs):
            self.linear_weights, f = optimizer.train(self.fi, self.linear_weights, self.Y)
            loss = optimizer.loss(self.fi, self.linear_weights, self.Y)
            print('loss:', loss, 'Residual error:', np.mean(np.abs(f-self.Y)))
            self.training_loss.append(loss)

    def predict(self, X, Y):
        fi = self.centroids.get_fi(X, self.sigma)
        _, y = self.optimizer.train(fi, self.linear_weights, Y)
        residual = np.mean(np.abs(y-Y))
        return np.dot(fi, self.linear_weights), residual

def test():

    pass


if __name__ == '__main__':
    test()
