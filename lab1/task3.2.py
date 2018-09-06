import numpy as np
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


def generate_weights(inputs, hidden_nodes):
    W = [np.random.normal(0, 0.001, (hidden_nodes, inputs.shape[0])),
         np.random.normal(0, 0.001, (1, hidden_nodes))]

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




def transfer(H):
    return (2 / (1+np.exp(-H))) - 1

def forward_pass(W, inputs):
    H = np.dot(W[0], inputs)
    H = transfer(H)
    O = np.dot(W[1], H)
    O = transfer(O)
    return O, H

def backward_pass(W, labels, O, H):
    dO = (O - labels) * ((1 + O) * (1 - O)) * 0.5
    dH = np.dot(W[1].T, dO) * ((1 + H) * (1 - H)) * 0.5
    return dO, dH

def update_weights(inputs, H, dO, dH, eta, W_momentum, use_momentum=False):

    alpha = 0.9

    if use_momentum:
        W_momentum = [(W_momentum[0] * alpha) - np.dot(dH, inputs.T) * (1 - alpha),
                      (W_momentum[1] * alpha) - np.dot(dO, H.T) * (1 - alpha)]
        dW = [W_momentum[0] * eta,
              W_momentum[1] * eta]
    else:
        dW = [-eta * np.dot(dH, inputs.T),
              -eta * np.dot(dO, H.T)]
    return dW, W_momentum

def compute_cost(O, labels):
    return np.sum((labels - O)**2)/2

def predict(W, inputs):
    _, O = forward_pass(W, inputs)
    return H

def perceptron(inputs, labels, W, epochs, eta, use_batch=True, use_momentum=False):

    plot_classes(inputs, labels)
    cost = []
    W_momentum = [np.zeros(W[0].shape), np.zeros(W[1].shape)]
    for i in range(epochs):
        O, H = forward_pass(W, inputs)
        dO, dH = backward_pass(W, labels, O, H)
        dW, W_momentum = update_weights(inputs, H, dO, dH, eta, W_momentum, use_momentum)
        W[0] += dW[0]
        W[1] += dW[1]
        print(compute_cost(O, labels))
        #plot_classes(inputs, labels)
        #draw_line(W)

plt.ion()
plt.show()

'''
NOTES:
 - Delta rule must use symmetric labels
 - Perceptron rule must use asymmetric labels
 - not using batch explodes with large learning rate
 - not using batch and no delta rule makes model wiggle
'''

inputs, labels = generate_binary_data(bias=True, symmetric_labels=False)
W = generate_weights(inputs, 50)

perceptron(inputs, labels, W, 100, 0.01, use_batch=True, use_momentum=False)

plt.show(block=True)
