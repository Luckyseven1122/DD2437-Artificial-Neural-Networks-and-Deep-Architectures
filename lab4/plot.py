import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

plt.ion()

class Plot():
    def __init__(self):
        pass

    def one(self, data):
        assert data.shape == (784,)
        plt.imshow(data.reshape(28,28))
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
    
    def get_img_matrix(self, data, rows, cols):
        render = np.zeros((rows*28, cols*28))
        for r in range(rows):
            row = np.zeros((28, cols*28))
            for c in range(cols):
                row[:,c*28:(c+1)*28] = data[c*rows+r].reshape(28,28)
            render[r*28:(r+1)*28,:] = row
        return render
    
    def custom(self, data, rows, cols, cost, epoch, i, eta, hidden_size):

        plt.clf()
        render = self.get_img_matrix(data, rows,cols)

        ax = plt.gca()
        fig = plt.gcf()
        title = 'Epoch:\t\t{0}\nCost:\t\t{1}\neta:\t\t{2}\n#hidden:   {3}'.format(epoch,cost,eta, hidden_size).expandtabs()
        fig.suptitle(title ,fontsize=24, x=0 , y=0.99, horizontalalignment='left')
        plt.imshow(render)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        plt.pause(1e-10)
        fig.savefig('gifs/tmp/'+str(i)+'.png')
        # plt.show()
