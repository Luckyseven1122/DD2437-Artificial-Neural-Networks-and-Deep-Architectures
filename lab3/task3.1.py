import numpy as np
import matplotlib.pyplot as plt


clean = np.array([[-1,-1, 1,-1, 1,-1,-1, 1],
                  [-1,-1,-1,-1,-1, 1,-1,-1],
                  [-1, 1, 1,-1,-1, 1,-1, 1]])

noisy = np.array([[ 1,-1, 1,-1, 1,-1,-1, 1],
                  [ 1, 1,-1,-1,-1, 1,-1,-1],
                  [ 1, 1, 1,-1, 1, 1,-1, 1]])


def little_model(X, epochs=1):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))
    for e in range(epochs):
        for i in range(P):
            x = X[i,:,None]
            # Calculate weights
            W += np.outer(x, x)
        for i in range(P):
            # Get next X
            x = X[i,:,None]
            X[i,:,None] = np.sign(np.dot(W.T, x))
    return W

def recall(data, W):
    P = data.shape[0]
    for i in range(P):
        x = np.sum(np.dot(W, data[i,:].reshape(1,-1).T), axis=1)
        print('data:  ',data[i,:])
        print('recall:', np.where(x > 0, 1, -1))


W = little_model(clean.copy())
recall(clean, W)
