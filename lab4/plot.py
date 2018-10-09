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
        for r in range(rows):
            row = np.zeros((28, cols*28))
            for c in range(cols):
                row[:,c*28:(c+1)*28] = data[c*rows+r].reshape(28,28)
            render[r*28:(r+1)*28,:] = row
        ax = plt.gca()
        plt.imshow(render)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        plt.show()

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
