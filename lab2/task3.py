import numpy as np
import matplotlib.pyplot as plt
from src.Network import Network
from src.Optimizer import LeastSquares
from src.Initializer import RandomNormal


def generate_data_task31(func):
    assert func == (np.sin or np.square)

    # training
    train_X = np.arange(0, 2*np.pi, 0.1).reshape(-1,1) # (n,1)
    train_Y = func(2*train_X)

    # testing
    test_X = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
    test_Y = func(2*test_X)

    return {'X': train_X, 'Y': train_Y}, {'X': test_X, 'Y': test_Y}


training, testing = generate_data_task31(np.sin)


RadialBasisNetwork = Network(X=training['X'],
                             Y=training['Y'],
                             hidden_nodes=30,
                             sigma=1.0,
                             initializer=RandomNormal())

RadialBasisNetwork.train(epochs=30,
                         optimizer=LeastSquares())
