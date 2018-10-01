import numpy as np
import matplotlib.pyplot as plt


data = np.array([[-1,-1, 1,-1, 1,-1,-1, 1],
                 [-1,-1,-1,-1,-1, 1,-1,-1],
                 [-1, 1, 1,-1,-1, 1,-1, 1]])

def little_model(data):

    P = data.shape[0] # P patterns
    N = data.shape[1] # N Units 
    W = np.zeros((N,N))
    for i in range(P):
        x = data[i,:].reshape(1,-1)
        W += np.outer(x.T, x)
    return W

def recall(data, W):
    P = data.shape[0]
    for i in range(P):
        x = np.sum(np.dot(W, data[i,:].reshape(1,-1).T), axis=1)
        print('data:  ',data[i,:])
        print('recall:', np.where(x > 0, 1, -1))

W = little_model(data)
recall(data, W)
