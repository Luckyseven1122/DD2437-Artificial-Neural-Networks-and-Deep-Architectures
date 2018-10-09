import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

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

    def custom(self, data, rows, cols):
        render = np.zeros((rows*28, cols*28))
        for c in range(cols):
            col = np.zeros((28*rows, 28))
            for r in range(rows):
                col[r*28:(r+1)*28,:] = data[c+r].reshape(28,28)
            render[:,c*28:(c+1)*28] = col
        ax = plt.gca()
        plt.imshow(render)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        plt.show()

        '''
        plt.figure(figsize= (rows,cols))
        gs = gridspec.GridSpec(rows, cols)
        gs.update(wspace=0.025, hspace=0.05)
        for i in range(rows*cols):
            ax = plt.subplot(gs[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_aspect('equal')
            ax.imshow(data[i].reshape(28,28))
        plt.show()
        '''
#        	fig, ax_buf = plt.subplots(rows,cols)

#         for i, ax_row in enumerate(ax_buf):
#             for j, ax in enumerate(ax_row):
#                 idx = j * rows + i
#                 ax.set_axis_off()
#                 ax.set_aspect('equal')
#                 if(idx > len(data) - 1): continue
#                 ax.imshow(data[idx].reshape(28,28))
#         fig.subplots_adjust(wspace=0, hspace=0)
#         plt.show()
