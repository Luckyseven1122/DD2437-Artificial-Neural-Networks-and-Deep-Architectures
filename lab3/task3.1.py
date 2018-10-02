import numpy as np
import itertools
import matplotlib.pyplot as plt


clean = np.array([[-1,-1, 1,-1, 1,-1,-1, 1],
                  [-1,-1,-1,-1,-1, 1,-1,-1],
                  [-1, 1, 1,-1,-1, 1,-1, 1]])

noisy = np.array([[ 1,-1, 1,-1, 1,-1,-1, 1],
                  [ 1, 1,-1,-1,-1, 1,-1,-1],
                  [ 1, 1, 1,-1, 1, 1,-1, 1]])

# clean = np.array(list(itertools.product([-1, 1], repeat=8)))

def little_model(X, epochs=30):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))

    for p in range(P):
        x = X[p,None,:]
        W += np.outer(x,x)
    return W
    # for e in range(epochs):
    #     for i in range(P):
    #         x = X[i,None,:]
    #         W += np.outer(x, x)
    #     W[np.diag_indices(N)] = 0
    #     for i in range(P):
    #         x = X[i,None,:]
    #         X[i,None,:] = np.sign(np.dot(x, W.T))
    # return W


def recall(data, W):
    P = data.shape[0]
    c = 0
    steps = 3
    for i in range(steps):
        data = np.sign(np.dot(data, W))

    return data.astype(int)


W = little_model(clean.copy(), epochs=2000)
ans = recall(noisy, W)
# print('org', clean)

diffing = 0
for n, c in zip(ans, clean):
    diff = sum(abs(n-c))
    print(n, c)
    if(diff > 0):
        diffing += 1
print('different: ', diffing)

# recall(data, W)

