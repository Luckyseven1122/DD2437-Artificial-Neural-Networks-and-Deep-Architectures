import numpy as np
import json
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
        self.optimizer = None
        self.initializer = initializer

        # Metrics placement
        self.training_loss = []
        self.validation_loss = []

    def _pack_data_object(self, epochs):
        '''
        pack data for analysis.
        '''
        data = {
            'config': json.dumps({'config': {
                'n samples': self.n_samples,
                'M input nodes': self.M_input_nodes,
                'N hidden nodes': self.N_hidden_nodes,
                'sigma': self.sigma,
                'learning rule': self.optimizer.__name__ if self.optimizer != None else 'No optimizer',
                'initializer': self.initializer.__name__ if self.initializer != None else 'No initializer',
                'epochs': epochs,
            }}, indent=2),
            't_loss': self.training_loss,
            'v_loss': self.validation_loss,
        }

        return data

    def train(self, epochs, optimizer):
        self.optimizer = optimizer
        self.epochs = epochs

        self.fi = self.centroids.get_fi(self.X, self.sigma)

        # Adjust batch for sample or batch learning based on optimizer
        batch_idx = [np.arange(0, self.n_samples)] if self.optimizer.__name__ == 'LeastSquares' else list(range(0,self.n_samples))
        for e in range(epochs):
            loss = 0
            for batch in batch_idx:
                self.linear_weights = optimizer.train(self.fi[batch,:], self.linear_weights, self.Y[batch,:])
                loss += optimizer.loss(self.fi[batch,:], self.linear_weights, self.Y[batch,:])
            self.training_loss.append(loss)
            print('loss:', loss)
        return self._pack_data_object(epochs)


    def predict(self, X, Y):
        pred_fi = self.centroids.get_fi(X, self.sigma)
        prediction = self.optimizer.output(pred_fi, self.linear_weights)
        residual = np.mean(np.abs(prediction-Y))
        return prediction, residual


def test():

    pass


if __name__ == '__main__':
    test()
