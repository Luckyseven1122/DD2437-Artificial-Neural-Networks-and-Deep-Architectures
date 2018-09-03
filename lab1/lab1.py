import numpy as np
import matplotlib.pyplot as plt



def generate_binary_data():
    '''
    Generates two classes of points
    Note: axis are set between -3 and 3 on both axis
    Note: Labels (-1, 1)
    '''
    n_points = 100
    mA = np.array([ 1.3, 0.5])
    mB = np.array([-1.3, -0.5])
    sigmaA = 0.5
    sigmaB = 0.5

    x = np.zeros([3, n_points*2])
    x[0,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[0]
    x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]
    x[2,:n_points] = -1
    x[0,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[0]
    x[1,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[1]
    x[2,n_points:] = 1

    # shuffle columns in x
    X = np.zeros([3, n_points*2])
    Y = np.zeros([n_points*2])
    idx = np.random.permutation(n_points*2)
    for i in idx:
        X[:2,i] = x[:2,i]
        X[2,i] = 1
        Y[i] = x[2,i]

    Y = Y.astype(int)
    return X, Y

def plot_classes(X, Y):
    # force axis for "real-time" update in learning step
    plt.axis([-3, 3, -3, 3])
    plt.scatter(X[0,:], X[1,:], c=Y)
    plt.show()



X, Y = generate_binary_data()


plot_classes(X, Y)
