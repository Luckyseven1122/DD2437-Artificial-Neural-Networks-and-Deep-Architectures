import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def generate_binary_data(n_points=50, linear=True, class_modifier=0):
    '''
    class_modifier = 0: no subsampling
    class_modifier = 1: remove random 25% from each class
    class_modifier = 2: remove 50% from classA (labels = -1)
    class_modifier = 3: remove 50% from classB (labels = 1 )
    class_modifier = 4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and
                        80% from classA(1,:)>0 (i.e x1 > 0)
    '''
    assert isinstance(n_points, int)
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

    if class_modifier == 1:
        idx = np.arange(math.floor(x.shape[1]/2))
        idxA = idx[:math.floor(x.shape[1]/4)]
        idxB = idx[math.floor(x.shape[1]/4):] # +1 ???
        np.random.shuffle(idxA)
        np.random.shuffle(idxB)
        idxA = idxA[:math.floor(x.shape[1]/8)]
        idxB = idxB[:math.floor(x.shape[1]/8)]
        idx = np.concatenate((idxA, idxB))
        x = np.delete(x, idx, axis=1)
    if class_modifier == 2:
        idx = np.arange(math.floor(x.shape[1]/2))
        np.random.shuffle(idx)
        idx = idx[:math.floor(x.shape[1]/4)]
        x = np.delete(x, idx, axis=1)
    if class_modifier == 3:
        idx = np.arange(math.floor(x.shape[1]/2),x.shape[1])
        np.random.shuffle(idx)
        idx = idx[:math.floor(x.shape[1]/4)]
        x = np.delete(x, idx, axis=1)
    if class_modifier == 4:
        idx = np.arange(math.floor(x.shape[1]/2))
        classA = x[:,idx]
        xless = np.where(classA[0,:] < 0, idx, -1)
        xmore = np.where(classA[0,:] >= 0, idx, -1)
        xless = xless[xless >= 0]
        xmore = xmore[xmore >= 0]
        np.random.shuffle(xless)
        np.random.shuffle(xmore)
        xless_size = xless.shape[0]
        xmore_size = xmore.shape[0]
        xless = xless[:math.floor(x.shape[1]*0.8)]
        xmore = xmore[:math.floor(x.shape[1]*0.2)]
        idx = np.concatenate((xless, xmore))
        x = np.delete(x, idx, axis=1)
    # shuffle columns in x
    inputs = np.zeros([2, x.shape[1]])
    labels = np.zeros([1, x.shape[1]])
    idx = np.random.permutation(x.shape[1])
    for i in idx:
        inputs[:2,i] = x[:2,idx[i]]
        labels[0,i] = x[2,idx[i]]
    labels = labels.astype(int)

    return inputs, labels

def generate_encoder_data(n_points=50):
    assert isinstance(n_points, int)
    inputs = -np.ones((8, n_points))
    idx = np.random.randint(7, size=n_points)
    for i in range(inputs.shape[1]):
        inputs[idx[i],i] = 1
    labels = inputs.copy()
    return inputs, labels


def generate_bell_function():
    x = np.reshape(np.arange(-5, 5, 0.5), (20, 1))
    y = np.reshape(np.arange(-5, 5, 0.5), (20, 1))
    xx, yy = np.meshgrid(x, y)
    z = np.dot(np.exp(-x * x * 0.1), np.exp(-y * y * 0.1).T) - 0.5


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(xx,yy,z)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    return xx.reshape(1, 400), z.reshape(1,400)


def split_data(inputs, labels, test_size, validation_size):

    if test_size + validation_size == 0:
         return {'inputs': inputs, 'labels': labels}, {'inputs': None  , 'labels': None  }, {'inputs': None  , 'labels': None  }

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


def plot_classes(inputs, labels):
    plt.grid(True)
    plt.scatter(inputs[0,:], inputs[1,:], c=labels[0,:])
    plt.show()
    plt.waitforbuttonpress()

def line(W, x):
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
    plt.clf()

    #ylabel = "error (MSE)" if delta_rule else "error (T/F-ratio)"
    #title += "Gradient ascend w/ batch" if use_batch else "w/o batch"

    x = np.arange(0, epochs)
    plt.plot(x, training_cost, 'r', label='training cost')
    if validation_cost:
        plt.plot(x, validation_cost, 'g', label='validation cost')
    #plt.title(title, fontsize=14)
    plt.xlabel('epochs', fontsize=12)
    plt.legend()
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
    #O = np.where(O > 0, 1, -1)
    #print("o", O)
    #print("labels", labels)
    return np.sum((labels - O)**2)/2

def predict(W, inputs):
    O, _ = forward_pass(W, inputs)
    O = np.where(O > 0, 1, 0)
    return O

def generate_weights(inputs, settings):
    if not settings['he_init']:
        W = [np.random.normal(0, 0.1, (settings['hidden_nodes'], inputs.shape[0] + 1)),
            np.random.normal(0, 0.1, (settings['output_dim'], settings['hidden_nodes'] + 1))]
    else:
        W = [np.random.normal(0, np.sqrt(2 / (inputs.shape[0]+1)), (settings['hidden_nodes'], inputs.shape[0] + 1)),
            np.random.normal(0, np.sqrt(2 / (settings['hidden_nodes'] + 1)), (settings['output_dim'], settings['hidden_nodes'] + 1))]
    return W


def perceptron(training, validation, test, settings):
    assert isinstance(settings['epochs'], int)
    assert isinstance(settings['eta'], float)
    assert isinstance(settings['hidden_nodes'], int)
    assert isinstance(settings['output_dim'], int)
    assert isinstance(settings['use_batch'], bool)
    assert isinstance(settings['use_momentum'], bool)
    assert isinstance(settings['he_init'], bool)
    assert settings['output_dim'] > 0
    assert settings['epochs'] > 0
    assert settings['eta'] > 0
    assert settings['hidden_nodes'] > 0

    training_cost = []
    validation_cost = []
    inputs = training['inputs']
    labels = training['labels']

    W = generate_weights(inputs, settings)
    W_momentum = [np.zeros(W[0].shape), np.zeros(W[1].shape)]
    for i in range(settings['epochs']):
        O, H = forward_pass(W, inputs)
        dO, dH = backward_pass(W, labels, O, H)
        dW, W_momentum = update_weights(inputs, H, dO, dH, settings['eta'], W_momentum, settings['use_momentum'])
        W[0] += dW[0]
        W[1] += dW[1]
        print("cost", compute_cost(O, labels))
        training_cost.append(compute_cost(O, labels))
        if isinstance(validation['inputs'], np.ndarray):
            _O, _ = forward_pass(W, validation['inputs'])
            print("validation", compute_cost(_O, validation['labels']))
            validation_cost.append(compute_cost(_O, validation['labels']))
    #plot_decision_boundary(inputs, lambda x: predict(W, x))
    #plot_classes(inputs, labels)
    plot_cost(training_cost, validation_cost, settings['epochs'], settings['use_batch'])

    # test
    if isinstance(test['inputs'], np.ndarray):
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

#inputs, labels = generate_binary_data(50, linear=False, class_modifier=4)




'''
# ENCODER PROBLEM SETUP
inputs, labels = generate_encoder_data(2000)
training, validation, test = split_data(inputs, labels, test_size=0.2, validation_size=0.4)

#
network_settings = {
    'epochs'       : 2000,
    'eta'          : 0.001,
    'hidden_nodes' : 3,
    'output_dim'   : 8,
    'use_batch'    : True,
    'use_momentum' : True,
    'he_init'      : True,
}
'''

inputs, labels = generate_bell_function()
training, validation, test = split_data(inputs, labels, test_size=0.2, validation_size=0.4)

network_settings = {
    'epochs'       : 100,
    'eta'          : 0.01,
    'hidden_nodes' : 500,
    'output_dim'   : 1,
    'use_batch'    : True,
    'use_momentum' : True,
    'he_init'      : False,
}

perceptron(training, validation, test, network_settings)
plt.show(block=True)
