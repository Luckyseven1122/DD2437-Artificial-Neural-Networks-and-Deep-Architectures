import numpy as np
from sympy.utilities.iterables import multiset_permutations
import matplotlib.pyplot as plt


clean = np.array([[-1,-1, 1,-1, 1,-1,-1, 1],
                  [-1,-1,-1,-1,-1, 1,-1,-1],
                  [-1, 1, 1,-1,-1, 1,-1, 1]])

noisy = np.array([[ 1,-1, 1,-1, 1,-1,-1, 1],
                  [ 1, 1,-1,-1,-1, 1,-1,-1],
                  [ 1, 1, 1,-1, 1, 1,-1, 1]])


# clean = np.array(list(itertools.product([-1, 1], repeat=8)))

def little_model(X):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))

    for p in range(P):
        x = X[p,None,:]
        W += np.outer(x,x)
    return W



def recall(data, W, steps):
    P = data.shape[0]
    for i in range(steps):
        data = np.sign(np.dot(data, W.T))
    return data.astype(int)


W = little_model(clean.copy())
ans = recall(noisy, W, steps=3)
# print('org', clean)

diffing = 0
for n, c in zip(ans, clean):
    print(n, c)
    if not ((n == c).all()):
        diffing += 1
print('different: ', diffing)

# recall(data, W)
