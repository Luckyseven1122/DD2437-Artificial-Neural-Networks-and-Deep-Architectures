import numpy as np
import matplotlib.pyplot as plt
from src.Network import Network
from src.Optimizer import LeastSquares, DeltaRule
from src.Initializer import RandomNormal
from src.Centroids import Fixed

def square(x):
    '''
    pass y = sin(2x)
    '''
    return np.where(x >= 0, 1, -1).astype(float)

def get_radial_coordinates():
    '''
    m = np.array([[np.pi/4  ],
                  [3*np.pi/4],
                  [5*np.pi/4],
                  [7*np.pi/4]]).T
    '''
    # Used for square(sin(2x))
    m = np.array([[ 1.4, 2.6, 3.4, 5.3]])

    # used for sin(2x)
    #m = np.arange(0.01, 2*np.pi - 0.01, 0.105).reshape(-1,1).T
    return m, m.shape[1]

def generate_data_task31(func, noise_std):

    X = np.arange(0, 2*np.pi, 0.01).reshape(-1,1)
    Y = func(2*X)

    if noise_std > 0:
        Y += np.random.normal(0, noise_std, Y.shape)

    train_X = X[::10].copy()
    train_Y = Y[::10].copy()

    test_X = X[5::10].copy()
    test_Y = Y[5::10].copy()

    return {'X': train_X, 'Y': train_Y}, {'X': test_X, 'Y': test_Y}


def task31():
    #training, testing = generate_data_task31(lambda x:np.sin(x), 0.1)
    training, testing = generate_data_task31(lambda x:square(np.sin(x)), 0.1)
    rbf_nodes, N_hidden_nodes = get_radial_coordinates()

    RadialBasisNetwork = Network(X=training['X'],
                                Y=training['Y'],
                                sigma=1.0,
                                hidden_nodes=N_hidden_nodes,
                                centroids=Fixed(rbf_nodes),
                                initializer=RandomNormal())

    RadialBasisNetwork.train(epochs=1,
                             optimizer=LeastSquares())

    prediction, residual_error = RadialBasisNetwork.predict(testing['X'], testing['Y'])

    print('N_hidden_nodes:',N_hidden_nodes)
    print('residual_error', residual_error)
    plt.plot(testing['X'], testing['Y'], label='True')
    plt.plot(testing['X'], prediction, label='Prediction')
    plt.ylabel('sign(sin(2x))')
    plt.xlabel('x')
    plt.scatter(rbf_nodes, np.zeros(rbf_nodes.size))
    plt.legend()
    plt.show()


def task32():
    training, testing = generate_data_task31(lambda x:square(np.sin(x)), 0.1)
    rbf_nodes, N_hidden_nodes = get_radial_coordinates()

    RadialBasisNetwork = Network(X=training['X'],
                                 Y=training['Y'],
                                 sigma=1.0,
                                 hidden_nodes=N_hidden_nodes,
                                 centroids=Fixed(rbf_nodes),
                                 initializer=RandomNormal())

    RadialBasisNetwork.train(epochs=1,
                             optimizer=DeltaRule(eta=0.1))

    prediction, residual_error = RadialBasisNetwork.predict(testing['X'], testing['Y'])

    '''
    print('N_hidden_nodes:',N_hidden_nodes)
    print('residual_error', residual_error)
    plt.plot(testing['X'], testing['Y'], label='True')
    plt.plot(testing['X'], prediction, label='Prediction')
    plt.ylabel('sign(sin(2x))')
    plt.xlabel('x')
    plt.scatter(rbf_nodes, np.zeros(rbf_nodes.size))
    plt.legend()
    plt.show()
    '''



task31()
#task32()
