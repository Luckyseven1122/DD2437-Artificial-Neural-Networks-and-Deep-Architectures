import numpy as np
import sys
from iohandler import load_data
import matplotlib.pyplot as plt



def generate_binary_data(bias = True, symmetric_labels=False):
    '''
    Generates two classes of points
    Note: axis are set between -3 and 3 on both axis
    Note: Labels (-1, 1)
    '''
    n_points = 100
    mA = np.array([ 1.3, 0.5])
    mB = np.array([-1.2, -0.5])
    sigmaA = 0.5
    sigmaB = 0.5

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


def generate_weights(inputs):
    print('inputs.shape',inputs.shape)
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

def draw_two_lines(W1, W2):
    x = [-4, 4]
    y1 = [line(W1, x[0]), line(W1, x[1])]
    y2 = [line(W2, x[0]), line(W2, x[1])]
    plt.plot(x, y1, 'y')
    plt.plot(x, y2, 'x')
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

def plot_cost_comparison(cost_batch,cost_seq, epochs, delta_rule):
    # hold figure until window close
    # plt.waitforbuttonpress()

    ylabel = "error (MSE)" if delta_rule else "error (T/F-ratio)"
    title = "Delta learning rule " if delta_rule else "Perceptron learning rule "
    title += "w/ batch and sequential"

    x = np.arange(0, epochs)
    plt.plot(x, cost_batch, 'r', label='cost batch')
    plt.plot(x, cost_seq, 'b', label='cost seq')

    plt.title(title, fontsize=14)
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
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

def seq_perceptron(W, inputs, labels, eta,delta_rule):
    error = 0
    for sample in inputs.T:
        outputs = activation(W, inputs)
        T = threshold(outputs)
        dW, e = update_weights(inputs, labels, W, T, eta, delta_rule)
        W += dW
        error += e
    c = compute_cost(e, delta_rule)
    print("Seq cost: ", c)
    return c, W 

def batch_perceptron(W, inputs, labels, eta, delta_rule):
    outputs = activation(W, inputs)
    T = threshold(outputs)
    dW, e = update_weights(inputs, labels, W, T, eta, delta_rule)
    c = compute_cost(e, delta_rule)
    print("Batch cost: ", c)
    return c, dW

def perceptron(inputs, labels, W, epochs, eta, delta_rule=False, use_batch=True, use_seq_batch=False):

    plot_classes(inputs, labels)
    cost = []
    W_batch = W.copy()
    W_seq = W.copy()
    cost_batch = [] 
    cost_seq = []

    for i in range(epochs):
        if use_seq_batch:
            c_batch, dW_batch = batch_perceptron(W_batch,inputs,labels,eta,delta_rule)
            c_seq, W_new = seq_perceptron(W_seq,inputs,labels,eta,delta_rule) 

            W_batch += dW_batch
            W_seq = W_new

            cost_batch.append(c_batch)
            cost_seq.append(c_seq)

            print('seq c: ', c_seq, ' batch c: ', c_batch)
        if use_batch:
            c, dW = batch_perceptron(W,inputs,labels,eta,delta_rule)
            W += dW
            cost.append(c)
        else:
            c, W_new = seq_perceptron(W,inputs,labels,eta,delta_rule) 
            W = W_new
            cost.append(c)

        plot_classes(inputs, labels)
        if(use_seq_batch):
            draw_two_lines(W_batch, W_seq)
        else:
            draw_line(W)

    if(use_seq_batch):
        plot_cost_comparison(cost_batch, cost_seq, epochs, delta_rule)
    else:
        plot_cost(cost, epochs, delta_rule, use_batch)
plt.ion()

'''
NOTES:
 - Delta rule must use symmetric labels
 - Perceptron rule must use asymmetric labels
 - not using batch explodes with large learning rate
 - not using batch and no delta rule makes model wiggle
'''

#inputs, labels = generate_binary_data(bias=True, symmetric_labels=False)
inputs, labels = load_data(sys.argv[1])
W = generate_weights(inputs)

perceptron(inputs, labels, W, 20, 0.0001, delta_rule=False, use_batch=False, use_seq_batch=False)


plt.show(block=True)
