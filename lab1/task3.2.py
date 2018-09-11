import numpy as np
import matplotlib.pyplot as plt
import math


def generate_binary_data(linear=True):


    n_points = 300

    if linear:
        '''
        Generates two linearly separable classes of points
        Note: axis are set between -3 and 3 on both axis
        Note: Labels (-1, 1)
        '''

        mA = np.array([ 1.5, 0.5])
        mB = np.array([-1.5, -0.5])
        sigmaA = 0.4
        sigmaB = 0.4

        x = np.zeros([3, n_points*2])
        x[0,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]
    else:
        '''
        Generates two non-linearly separable classes of points
        '''
        mA = [ 1.0, 0.3]
        mB = [ 0.0, 0.0]
        sigmaA = 0.3
        sigmaB = 0.3

        x = np.zeros([3, n_points*2])
        x[0,:math.floor(n_points/2)] = np.random.randn(1, math.floor(n_points/2)) * sigmaA - mA[0]
        x[0,math.floor(n_points/2):n_points] = np.random.randn(1, math.floor(n_points/2)) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]
    x[2,:n_points] = -1
    x[0,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[0]
    x[1,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[1]
    x[2,n_points:] = 1

    # shuffle columns in x
    inputs = np.zeros([2, n_points*2])
    labels = np.zeros([1, n_points*2])
    idx = np.random.permutation(n_points*2)
    for i in idx:
        inputs[:2,i] = x[:2,idx[i]]
        labels[0,i] = x[2,idx[i]]
    labels = labels.astype(int)

    return inputs, labels

def modify_data(inputs, labels, class_modifier):
    '''
    class_modifier = 1: remove random 25% from each class
    class_modifier = 2: remove 50% from classA (labels = -1)
    class_modifier = 3: remove 50% from classB (labels = 1 )
    class_modifier = 4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and
                        80% from classA(1,:)>0 (i.e x1 > 0)
    '''
    n_samples = inputs.shape[1]

    if class_modifier == 1:
        classA = np.where(labels < 0, -1, 1)
        idx = np.random.randint(n_samples/2, size=int((n_samples/2)*0.25))
        #TODO:








def split_data(inputs, labels, test_size, validation_size):
    assert validation_size + test_size < 1
    validation_size = int(inputs.shape[1]*validation_size)
    test_size = int(inputs.shape[1]*test_size)

    validation = {'inputs': inputs[:,0:validation_size],
                  'labels': labels[:,0:validation_size]}
    test = {'inputs': inputs[:,validation_size+1:validation_size+test_size],
            'labels': labels[:,validation_size+1:validation_size+test_size]}
    training = {'inputs': inputs[:,validation_size+test_size+1:],
                'labels': labels[:,validation_size+test_size+1:]}

    print()
    print("Sample size:", inputs.shape[1])
    print("training set: in=", training['inputs'].shape[1], " lbl=", training['labels'].shape[1])
    print("validation set: in=", validation['inputs'].shape[1], " lbl=", validation['labels'].shape[1])
    print("test set: in=", test['inputs'].shape[1], " lbl=", test['labels'].shape[1])
    print()

    return training, validation, test

def generate_weights(inputs, hidden_nodes, he=True):

    if not he:
        W = [np.random.normal(0, 0.1, (hidden_nodes, inputs.shape[0] + 1)),
            np.random.normal(0, 0.1, (1, hidden_nodes + 1))]
    else:
        W = [np.random.normal(0, np.sqrt(2 / (inputs.shape[0]+1)), (hidden_nodes, inputs.shape[0] + 1)),
            np.random.normal(0, np.sqrt(2 / (hidden_nodes + 1)), (1, hidden_nodes + 1))]

    # Set bias weight to zero
    #if inputs[2,0] == 0:
    #   W[0,2] = 0
    return W

def plot_classes(inputs, labels):
    # force axis for "real-time" update in learning step
    #plt.clf()
    #plt.axis([-3, 3, -3, 3])
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

def plot_cost(training_cost, validation_cost, epochs, use_batch):
    # hold figure until window close
    plt.waitforbuttonpress()

    #ylabel = "error (MSE)" if delta_rule else "error (T/F-ratio)"
    #title += "Gradient ascend w/ batch" if use_batch else "w/o batch"

    x = np.arange(0, epochs)
    plt.plot(x, training_cost, 'r')
    plt.plot(x, validation_cost, 'g')
    #plt.title(title, fontsize=14)
    plt.xlabel('epochs', fontsize=12)
    #plt.ylabel(ylabel, fontsize=12)
    plt.show()


def plot_decision_boundary(X, predict):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - .5, X[0, :].max() + .5
    y_min, y_max = X[1, :].min() - .5, X[1, :].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    #plt.scatter(X[:, 0], X[:, 1], c=[0,1], cmap=plt.cm.Spectral)
    #plt.waitforbuttonpress()


def transfer(H):
    return (2 / (1+np.exp(-H))) - 1

def forward_pass(W, inputs):
    H = np.dot(W[0], np.concatenate((inputs, np.ones((1, inputs.shape[1])))))
    H = np.concatenate((transfer(H), np.ones((1, inputs.shape[1]))))
    O = np.dot(W[1], H)
    O = transfer(O)
    return O, H

def backward_pass(W, labels, O, H):
    dO = (O - labels) * ((1 + O) * (1 - O)) * 0.5
    dH = np.dot(W[1].T, dO) * ((1 + H) * (1 - H)) * 0.5
    dH = np.delete(dH, dH.shape[0]-1, 0)
    return dO, dH

def update_weights(inputs, H, dO, dH, eta, W_momentum, use_momentum=False):
    alpha = 0.9
    if use_momentum:
        W_momentum = [(W_momentum[0] * alpha) - np.dot(dH, np.concatenate((inputs, np.ones((1, inputs.shape[1])))).T) * (1 - alpha),
                      (W_momentum[1] * alpha) - np.dot(dO, H.T) * (1 - alpha)]
        dW = [W_momentum[0] * eta,
              W_momentum[1] * eta]
    else:
        dW = [-eta * np.dot(dH, np.concatenate((inputs, np.ones((1, inputs.shape[1])))).T),
              -eta * np.dot(dO, H.T)]
    return dW, W_momentum

def compute_cost(O, labels):
    #O = np.where(O > 0, 1, 0)
    #print("o", O)
    #print("labels", labels)
    return np.sum((labels - O)**2)/2

def predict(W, inputs):
    O, _ = forward_pass(W, inputs)
    O = np.where(O > 0, 1, 0)
    return O

def perceptron(training, validation, test, W, epochs, eta, use_batch=True, use_momentum=False):
    training_cost = []
    validation_cost = []

    inputs = training['inputs']
    labels = training['labels']

    W_momentum = [np.zeros(W[0].shape), np.zeros(W[1].shape)]
    for i in range(epochs):
        O, H = forward_pass(W, inputs)
        dO, dH = backward_pass(W, labels, O, H)
        dW, W_momentum = update_weights(inputs, H, dO, dH, eta, W_momentum, use_momentum)
        W[0] += dW[0]
        W[1] += dW[1]
        print("cost", compute_cost(O, labels))
        training_cost.append(compute_cost(O, labels))
        _O, _ = forward_pass(W, validation['inputs'])
        print("validation", compute_cost(_O, validation['labels']))
        validation_cost.append(compute_cost(_O, validation['labels']))

        #plot_classes(inputs, labels)
        #draw_line(W)
    plot_decision_boundary(inputs, lambda x: predict(W, x))
    plot_classes(inputs, labels)
    plot_cost(training_cost, validation_cost, epochs, use_batch)

    # test
    o, _ = forward_pass(W, test['inputs'])
    print("Test cost:", compute_cost(o, test['labels']))
plt.ion()
plt.show()

'''
NOTES:
 - Delta rule must use symmetric labels
 - Perceptron rule must use asymmetric labels
 - not using batch explodes with large learning rate
 - not using batch and no delta rule makes model wiggle
'''
inputs, labels = generate_binary_data(linear=False)
inputs, labels = modify_data(inputs, labels, 1)
training, validation, test = split_data(inputs, labels, test_size=0.2, validation_size=0.4)
W = generate_weights(training['inputs'], 50, he=True)

perceptron(training, validation, test, W, 1000, 0.01, use_batch=True, use_momentum=True)

plt.show(block=True)
