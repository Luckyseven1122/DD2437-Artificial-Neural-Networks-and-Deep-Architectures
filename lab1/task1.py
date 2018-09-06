import numpy as np
import math
import matplotlib.pyplot as plt



def generate_binary_data(bias = True, symmetric_labels=False, linear=True):


    if linear:
        '''
        Generates two linearly separable classes of points
        Note: axis are set between -3 and 3 on both axis
        Note: Labels (-1, 1)
        '''
        n_points = 100
        mA = np.array([ 1.0, 0.5])
        mB = np.array([-1.0, -0.5])
        sigmaA = 0.4
        sigmaB = 0.4

        x = np.zeros([3, n_points*2])
        x[0,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]
        x[2,:n_points] = -1 if symmetric_labels==True else 0
        x[0,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[0]
        x[1,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[1]
        x[2,n_points:] = 1

        # shuffle columns in x
        inputs = np.zeros([3, n_points*2])
        labels = np.zeros([1, n_points*2])
        idx = np.random.permutation(n_points*2)
        for i in idx:
            inputs[:2,i] = x[:2,idx[i]]
            inputs[2,i] = 1 if bias == True else 0 # used later on as bias term multipyer
            labels[0,i] = x[2,idx[i]]

        labels = labels.astype(int)

        return inputs, labels
    else:
        '''
        Generates two non-linearly separable classes of points
        '''
        n_points = 100
        mA = [ 1.0, 0.3]
        mB = [ 0.0, -0.1]
        sigmaA = 0.2
        sigmaB = 0.3

        x = np.zeros([3, n_points*2])
        x[0,:math.floor(n_points/2)] = np.random.randn(1, math.floor(n_points/2)) * sigmaA - mA[0]
        x[0,math.floor(n_points/2):n_points] = np.random.randn(1, math.floor(n_points/2)) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]
        x[2,:n_points] = -1 if symmetric_labels==True else 0
        x[0,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[0]
        x[1,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[1]
        x[2,n_points:] = 1

        # shuffle columns in x
        inputs = np.zeros([3, n_points*2])
        labels = np.zeros([1, n_points*2])
        idx = np.random.permutation(n_points*2)
        for i in idx:
            inputs[:2,i] = x[:2,idx[i]]
            inputs[2,i] = 1 if bias == True else 0 # used later on as bias term multipyer
            labels[0,i] = x[2,idx[i]]

        labels = labels.astype(int)

        return inputs, labels


def generate_weights(inputs):
    W = np.random.normal(0, 0.001, (1, inputs.shape[0]))
    # Set bias weight to zero
    if inputs[2,0] == 0:
       W[0,2] = 0
    return W

def plot_classes(inputs, labels):
    # force axis for "real-time" update in learning step
    plt.clf()
    plt.axis([-3, 3, -3, 3])
    plt.grid(True)
    plt.scatter(inputs[0,:], inputs[1,:], c=labels[0,:])
    plt.show()


def line(W, x):
    #k = -(W.T[2]/W.T[1])/(W.T[2]/W.T[0])
    k = -(W.T[0]/W.T[1])
    m = -W.T[2]/W.T[1]
    return k*x+m

def draw_line(W):
    x = [-4, 4]
    y = [line(W, x[0]), line(W, x[1])]
    plt.plot(x, y)
    plt.pause(0.01)
    plt.show()

def plot_cost(cost, epochs, delta_rule, use_batch):
    # hold figure until window close
    plt.waitforbuttonpress()

    ylabel = "error (MSE)" if delta_rule else "error (T/F-ratio)"
    title = "Delta learning rule " if delta_rule else "Perceptron learning rule "
    title += "w/ batch" if use_batch else "w/o batch"

    x = np.arange(0, epochs)
    plt.plot(x, cost, 'r')
    plt.title(title, fontsize=14)
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()


def activation(W, inputs):
    return np.dot(W, inputs)


def threshold(outputs):
    return np.where(outputs > 0, 1, 0)

def update_weights(inputs, labels, W, T, eta, delta_rule):
    '''
    Update weights
    '''
    if delta_rule:
        e = labels - activation(W, inputs)
    else:
        e = labels - T
    return eta*np.dot(e, inputs.T), e

def compute_cost(errors, delta_rule):
    if delta_rule:
        return np.mean(errors**2) # mse
    else:
        return np.where((errors) == 0, 0, 1).mean() # ratio

def perceptron(inputs, labels, W, epochs, eta, delta_rule=False, use_batch=True):

    plot_classes(inputs, labels)
    cost = []
    for i in range(epochs):
        if use_batch:
            outputs = activation(W, inputs)
            T = threshold(outputs)
            dW, e = update_weights(inputs, labels, W, T, eta, delta_rule)
            W += dW
            c = compute_cost(e, delta_rule)
            cost.append(c)
            print(c)
        else:
            error = 0
            for sample in inputs.T:
                outputs = activation(W, inputs)
                T = threshold(outputs)
                dW, e = update_weights(inputs, labels, W, T, eta, delta_rule)
                W += dW
                error += e
            c = compute_cost(e, delta_rule)
            cost.append(c)
            print(c)
        plot_classes(inputs, labels)
        draw_line(W)
    plot_cost(cost, epochs, delta_rule, use_batch)
plt.ion()
plt.show()

'''
NOTES:
 - Delta rule must use symmetric labels
 - Perceptron rule must use asymmetric labels
 - not using batch explodes with large learning rate
 - not using batch and no delta rule makes model wiggle
'''

inputs, labels = generate_binary_data(bias=True, symmetric_labels=False, linear=False)
W = generate_weights(inputs)

perceptron(inputs, labels, W, 35, 0.0001, delta_rule=False, use_batch=True)


plt.show(block=True)
