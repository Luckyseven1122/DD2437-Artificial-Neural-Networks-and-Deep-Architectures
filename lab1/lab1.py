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
    mB = np.array([-1.5, -0.5])
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
    Y = np.zeros([1, n_points*2])
    idx = np.random.permutation(n_points*2)
    for i in idx:
        X[:2,i] = x[:2,i]
        X[2,i] = 1 # used later on as bias term multipyer
        Y[0,i] = x[2,i]

    Y = Y.astype(int)
    return X, Y

def plot_classes(X, Y):
    # force axis for "real-time" update in learning step
    plt.axis([-3, 3, -3, 3])
    plt.scatter(X[0,:], X[1,:], c=Y[0,:])
    plt.show()


def line(W, x):
    k = -(W.T[2]/W.T[1])/(W.T[2]/W.T[0])
    m = -W.T[2]/W.T[1]
    return k*x+m

def draw_line(W, X):
    x = [-4, 4]
    y = [line(W, x[0]), line(W, x[1])]
    plt.plot(x, y)





def generate_weights(X):
    W = np.random.normal(0, 0.01, (1, X.shape[0]))
    return W


X, Y = generate_binary_data()
W = generate_weights(X)


'''
Perceptron Learning
'''

def compute_cost(W, X, Y):
    T = np.dot(W, X)
    T = np.where(T > 0, -1, 1)
    T = np.where((Y-T) != 0, 1, 0)
    return np.mean(T)



def weigth_update(X, Y, W, T, eta, delta_rule=False):
    if not delta_rule:
        dW = eta * np.dot(T-Y, X.T)
        #dW = -eta * np.dot(Y-T, X.T)
    else:
        e = T - np.dot(W, X)
        dW = eta * np.dot(e, X.T)
        #dW = -eta * np.dot((np.dot(W, X) - T), X.T)
    return dW

def perceptron(X, Y, W, eta, n_epochs, delta_rule=False, use_batch=True):

    for i in range(0, n_epochs):
        if use_batch:
            T = np.dot(W, X)
            T = np.where(T > 0, -1, 1)
            W += weigth_update(X, Y, W, T, eta, delta_rule)
        else:
            for i in range(X.shape[1]):
                x = X[:,i,None]
                y = Y[:,i,None]
                T = np.dot(W, x)
                T = np.where(T > 0, -1, 1)
                W += weigth_update(x, y, W, T, eta, delta_rule)
        print("Error:", compute_cost(W, X, Y))
        draw_line(W, X)
        plot_classes(X, Y)


perceptron(X, Y, W, 0.01, 5, delta_rule=False, use_batch=False)
