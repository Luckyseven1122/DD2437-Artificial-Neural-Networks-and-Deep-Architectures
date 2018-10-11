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

#     def custom(self, data, rows, cols):
#         gs = gridspec.GridSpec(rows, cols)
#         gs.update(wspace=0, hspace=0)
#         for i in range(rows*cols):
#             ax = plt.subplot(gs[i])
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             ax.set_aspect('equal')
#             ax.imshow(data[i].reshape(28,28))
#         plt.pause(1e-12)

    def loss(self, loss, label, save=None):
        plt.clf()
        x = np.arange(len(loss))
        plt.plot(x, loss, label=label)
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.show()

    def custom(self, data, rows, cols, save=None):
        plt.clf()
        render = np.zeros((rows*28, cols*28))
        for r in range(rows):
            row = np.zeros((28, cols*28))
            for c in range(cols):
                row[:,c*28:(c+1)*28] = data[c*rows+r].reshape(28,28)
            render[r*28:(r+1)*28,:] = row
        ax = plt.gca()
        plt.imshow(render, cmap='gray')
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.pause(1e-12)
        # plt.show()

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
                plt.plot(x, loss, label=labels[i])
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        if labels != None:
            plt.legend()
        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.show()
