import numpy as np
import json
from .Initializer import Initializer
from .Centroids import Fixed
import matplotlib.pyplot as plt

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

        self.X = X.copy()
        self.Y = Y.copy()

        self.sigma = sigma

        # extract dimensionality information
        self.n_samples = X.shape[0]
        self.M_input_nodes = X.shape[1]
        self.N_hidden_nodes = hidden_nodes
        self.O_output_size = Y.shape[1]

        # Ensure N < n
        assert self.N_hidden_nodes < self.n_samples

        self.fi = None
        self.centroids = centroids
        self.centroids.set_dims(N=self.N_hidden_nodes, M=self.M_input_nodes, O=self.O_output_size)
        self.linear_weights = initializer.new((self.N_hidden_nodes, self.O_output_size))
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
                'epoch_shuffle': self.epoch_shuffle,
                'centroid eta': self.centroids._eta,
            }}, indent=2),
            't_loss': self.training_loss,
            'v_loss': self.validation_loss,
        }

        return data

    def train(self, epochs, optimizer, epoch_shuffle=False):
        self.optimizer = optimizer
        self.optimizer.set_dims(N=self.N_hidden_nodes, M=self.M_input_nodes, O=self.O_output_size)
        self.epochs = epochs
        self.epoch_shuffle = epoch_shuffle



        # Adjust batch for sample or batch learning based on optimizer
        batch_idx = [np.arange(0, self.n_samples)] if self.optimizer.__name__ == 'LeastSquares' else list(range(0,self.n_samples))
        #plt.axis([-0.01, 1, -0.01, 1])
        #plt.scatter(self.Y[:,0], self.X[:,0], marker='x', label='angle/distances', alpha=0.7)
        #plt.scatter(self.Y[:,1], self.X[:,1], marker='x', label='velocity/height', alpha=0.7)
        for e in range(epochs):
            loss = 0

            if epoch_shuffle:
                idx = np.random.permutation(np.arange(self.n_samples))
                self.X = self.X[idx]
                self.Y = self.Y[idx] # used or no?

            self.fi = self.centroids.get_fi(self.X, self.sigma)

            for batch in batch_idx:
                self.linear_weights = optimizer.train(self.fi[batch,:], self.linear_weights, self.Y[batch,:])
                loss += optimizer.loss(self.fi[batch,:], self.linear_weights, self.Y[batch,:])
            self.training_loss.append(loss)
            print('loss:', loss)

            #if (e % 10) == 0:
            #    m = self.centroids.get_matrix()
            #    plt.scatter(m[0,:], m[1,:], color='blue', alpha=0.2)

        #m = self.centroids.get_matrix()
        #plt.scatter(m[0,:], m[1,:], color='blue', label='RBF Node')
        #fig = plt.gcf()
        #ax = fig.gca()
        #for i in range(m.shape[1]):
        #    circle = plt.Circle((m[0,i], m[1,i]), self.sigma, color='r')
        #    circle.set_clip_box(ax.bbox)
        #    circle.set_edgecolor( 'r' )
        #    circle.set_facecolor( 'none' )
        #    circle.set_alpha( 0.5 )
        #    ax.add_artist(circle)
        #plt.legend()
        #plt.show()

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
