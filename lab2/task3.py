import numpy as np
import matplotlib.pyplot as plt
from src.Network import Network
from src.Optimizer import LeastSquares
from src.Initializer import RandomNormal
from src.Centroids import Fixed

def get_radial_coordinates(arg):
    if arg == 1:
        m = np.array([[np.pi/4  ],
                      [3*np.pi/4],
                      [5*np.pi/4],
                      [7*np.pi/4]])

    return m.T, m.shape[0]

def generate_data_task31(func):
    # training
    train_X = np.arange(0, 2*np.pi, 0.1).reshape(-1,1) # (n,1)
    train_Y = func(2*train_X)

    # testing
    test_X = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
    test_Y = func(2*test_X)


    return {'X': train_X, 'Y': train_Y}, {'X': test_X, 'Y': test_Y}


training, testing = generate_data_task31(np.sin)
rbf_nodes, n_nodes = get_radial_coordinates(1)

plt.plot(training['X'], training['Y'])
plt.show()



RadialBasisNetwork = Network(X=training['X'],
                             Y=training['Y'],
                             sigma=1.0,
                             hidden_nodes=n_nodes,
                             centroids=Fixed(rbf_nodes),
                             initializer=RandomNormal())

RadialBasisNetwork.train(epochs=30,
                         optimizer=LeastSquares())
