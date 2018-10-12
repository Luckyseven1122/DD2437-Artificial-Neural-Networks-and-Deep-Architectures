import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
plt.ion()

class Plot():
    def __init__(self):
        pass

    def one(self, data):
        assert data.shape == (784,)
        plt.imshow(data.reshape(28,28), cmap='gray')
        plt.show()

    def four(self, data):
        assert data.shape == (4,784)

        self.custom(data, 2,2)

    def nine(self, data):
        assert data.shape == (9, 784)
        self.custom(data, 3,3)

    def loss(self, loss):
        x = np.arange(len(loss))
        plt.plot(x, loss)
        plt.show()

    def custom(self, data, rows, cols, save=None, dims=(32,16)):
        plt.clf()
        render = np.zeros((rows*dims[0], cols*dims[1]))
        for r in range(rows):
            row = np.zeros((dims[0], cols*dims[1]))
            for c in range(cols):
                row[:,c*dims[1]:(c+1)*dims[1]] = data[c*rows+r].reshape(dims)
            render[r*dims[0]:(r+1)*dims[0],:] = row
        ax = plt.gca()
        plt.imshow(render, cmap='gray')
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.pause(1e-12)
        plt.show(block=True)

    def losses(self, loss_list, labels=None, save=None):
        '''
        labels = ['str','in','gs'] with len of loss_list
        '''
        plt.clf()
        for i, loss in enumerate(loss_list):
            x = np.arange(len(loss))
            if labels == None:
                plt.plot(x, loss)
            else:
                assert len(loss_list) == len(labels)
                style = labels[i].split(' ')[0]
                if(style == 'test'):
                    plt.plot(x, loss, label=labels[i])
                else:
                    plt.plot(x, loss, label=labels[i], linestyle='dashed')
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        if labels != None:
            plt.legend()
        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.show()
'''
plot = Plot()
cost_layer_500_30_test = np.load('cost_layer_500_30_test.npy')
cost_layer_500_50_test = np.load('cost_layer_500_50_test.npy')
cost_layer_500_100_test = np.load('cost_layer_500_100_test.npy')
cost_layer_500_250_test = np.load('cost_layer_500_250_test.npy')

cost_layer_500_30_train = np.load('cost_layer_500_30_train.npy')
cost_layer_500_50_train = np.load('cost_layer_500_50_train.npy')
cost_layer_500_100_train = np.load('cost_layer_500_100_train.npy')
cost_layer_500_250_train = np.load('cost_layer_500_250_train.npy')

data = [cost_layer_500_30_test, cost_layer_500_50_test, cost_layer_500_100_test, cost_layer_500_250_test,
        cost_layer_500_30_train, cost_layer_500_50_train, cost_layer_500_100_train, cost_layer_500_250_train]

test_labels = ['test N=30','test N=50', 'test N=100', 'test N=250','train N=30','train N=50', 'train N=100', 'train N=250']
plot.losses(data, test_labels)

plt.show(block=True)
'''
