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
        colors = ['b', 'g', 'r', 'c']
        plt.clf()
        for i, loss in enumerate(loss_list):
            x = np.arange(len(loss))
            print(i % 4)
            if labels == None:
                plt.plot(x, loss)
            else:
                assert len(loss_list) == len(labels)
                style = labels[i].split(' ')[0]
                if(style == 'test'):
                    plt.plot(x, loss, label=labels[i], color=colors[(i) % 4]) 
                else:
                    plt.plot(x, loss, label=labels[i], linestyle='dashed', color=colors[(i) % 4]) 
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        if labels != None:
            plt.legend()
        if save != None:
            plt.savefig('./tmp/' + str(save['path']) + '.png')
        else:
            plt.show()

    def loss(self, loss_test, loss_train, N):
        x = np.arange(len(loss_test))
        test_label = 'test'
        train_label = 'train'
        plt.plot(x,loss_test, label=test_label)
        plt.plot(x,loss_train, label=train_label)
        plt.xlabel('epochs', fontsize=20)
        plt.ylabel('mean squared error', fontsize=20)
        plt.legend(prop={'size': 20})
        # plt.show()


plot = Plot()
cost_layer_500_50_40_test = np.load('cost_layer_500_50_40_test.npy')
cost_layer_500_50_30_test = np.load('cost_layer_500_50_30_test.npy')
cost_layer_500_50_20_test = np.load('cost_layer_500_50_20_test.npy')
cost_layer_500_50_15_test = np.load('cost_layer_500_50_15_test.npy')

cost_layer_500_50_40_train = np.load('cost_layer_500_50_40_train.npy')
cost_layer_500_50_30_train = np.load('cost_layer_500_50_30_train.npy')
cost_layer_500_50_20_train = np.load('cost_layer_500_50_20_train.npy')
cost_layer_500_50_15_train = np.load('cost_layer_500_50_15_train.npy')

cost_layer_500_50_train_BEST = np.load('cost_layer_500_50_15_train_BEST.npy')
cost_layer_500_50_test_BEST = np.load('cost_layer_500_50_15_test_BEST.npy')

plot.loss(cost_layer_500_50_test_BEST, cost_layer_500_50_train_BEST, 50)
plt.title('MSE for Network setup: 784 > 500 > 50 > 10, Accuracy of 89 %', fontsize=24, y=1.01)


# plt.subplot(221)
# plot.loss(cost_layer_500_50_15_test, cost_layer_500_50_15_train, 15)

# axes = plt.gca()
# axes.set_xlim([0,200])
# axes.set_ylim([0,6])

# plt.subplot(222)
# plot.loss(cost_layer_500_50_20_test, cost_layer_500_50_20_train, 20)

# axes = plt.gca()
# axes.set_xlim([0,200])
# axes.set_ylim([0,6])

# plt.subplot(223)
# plot.loss(cost_layer_500_50_30_test, cost_layer_500_50_30_train, 30)

# axes = plt.gca()
# axes.set_xlim([0,200])
# axes.set_ylim([0,6])

# plt.subplot(224)
# plot.loss(cost_layer_500_50_40_test, cost_layer_500_50_40_train, 40)

# plt.suptitle('MSE error with different number of hidden nodes in the third layer.', fontsize=26, y=0.93)

# axes = plt.gca()
# axes.set_xlim([0,200])
# axes.set_ylim([0,6])

# plt.show(block=True)

# PLOTS FOR 784 500 > 30, 50, 100, 250
#plot = Plot()
#cost_layer_500_30_test = np.load('cost_layer_500_30_test.npy')
#cost_layer_500_50_test = np.load('cost_layer_500_50_test.npy')
#cost_layer_500_100_test = np.load('cost_layer_500_100_test.npy')
#cost_layer_500_250_test = np.load('cost_layer_500_250_test.npy')
#
#cost_layer_500_30_train = np.load('cost_layer_500_30_train.npy')
#cost_layer_500_50_train = np.load('cost_layer_500_50_train.npy')
#cost_layer_500_100_train = np.load('cost_layer_500_100_train.npy')
#cost_layer_500_250_train = np.load('cost_layer_500_250_train.npy')
#
#
#plt.subplot(221)
#plot.loss(cost_layer_500_30_test, cost_layer_500_30_train, 30)
#
#plt.subplot(222)
#plot.loss(cost_layer_500_50_test, cost_layer_500_50_train, 50)
#
#plt.subplot(223)
#plot.loss(cost_layer_500_100_test, cost_layer_500_100_train, 100)
#
#plt.subplot(224)
#plot.loss(cost_layer_500_250_test, cost_layer_500_250_train, 250)
#
#plt.suptitle('MSE error with different number of hidden nodes in the second layer.', fontsize=26, y=0.93)
plt.show(block=True)
