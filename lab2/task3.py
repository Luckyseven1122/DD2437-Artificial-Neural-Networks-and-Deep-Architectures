import numpy as np
import matplotlib.pyplot as plt
from src.Network import Network
from src.Optimizer import LeastSquares
from src.Initializer import RandomNormal
from src.Centroids import Fixed

def square(x):
    '''
    pass y = sin(2x)
    '''
    return np.where(x >= 0, 1, -1)

def get_radial_coordinates():
    '''
    m = np.array([[np.pi/4  ],
                  [3*np.pi/4],
                  [5*np.pi/4],
                  [7*np.pi/4]]).T
    '''
    m = np.array([[ 1.4, 2.6, 3.4, 5.3]])

    #m = np.arange(0.01, 2*np.pi - 0.01, 0.105).reshape(-1,1).T
    return m, m.shape[1]

def generate_data_task31(func):
    # training
    train_X = np.arange(0, 2*np.pi, 0.1).reshape(-1,1) # (n,1)
    train_Y = func(2*train_X)

    # testing
    test_X = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
    test_Y = func(2*test_X)


    return {'X': train_X, 'Y': train_Y}, {'X': test_X, 'Y': test_Y}


training, testing = generate_data_task31(lambda x: square(np.sin(x)))
rbf_nodes, N_hidden_nodes = get_radial_coordinates()

#plt.plot(training['X'], training['Y'])
#plt.show()


RadialBasisNetwork = Network(X=training['X'],
                             Y=training['Y'],
                             sigma=1.0,
                             hidden_nodes=N_hidden_nodes,
                             centroids=Fixed(rbf_nodes),
                             initializer=RandomNormal())

RadialBasisNetwork.train(epochs=1,
                         optimizer=LeastSquares())

print('N_hidden_nodes:',N_hidden_nodes)

prediction, residual_error = RadialBasisNetwork.predict(testing['X'], testing['Y'])

print('residual_error', residual_error)
plt.plot(testing['X'], testing['Y'], label='True')
plt.plot(testing['X'], np.where(prediction >= 0, 1, -1), label='Prediction')
plt.ylabel('sign(sin(2x))')
plt.xlabel('x')
plt.scatter(rbf_nodes, np.zeros(rbf_nodes.size))
plt.legend()
plt.show()
