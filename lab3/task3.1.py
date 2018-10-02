import numpy as np
import matplotlib.pyplot as plt
import json

clean = np.array([[-1,-1, 1,-1, 1,-1,-1, 1],
                  [-1,-1,-1,-1,-1, 1,-1,-1],
                  [-1, 1, 1,-1,-1, 1,-1, 1]])

noisy = np.array([[ 1,-1, 1,-1, 1,-1,-1, 1],
                  [ 1, 1,-1,-1,-1, 1,-1,-1],
                  [ 1, 1, 1,-1, 1, 1,-1, 1]])

noisy2 = np.array([[-1,1, -1,1, -1,1,-1, 1],
                   [1,1,-1,1,-1, 1,1,1],
                   [1, -1, -1,1,-1, -1,-1, 1]])


def int_to_array(number):
    '''
    converts numbers in range 0 to 255 into lists
    '''
    assert number < 256 and number >= 0
    res = []
    for i in range(8):
        res.append(1)
        if (number & 128) == 0:
            res[-1] = -1
        number = number << 1
    return np.array(res).reshape(1,-1)

# clean = np.array(list(itertools.product([-1, 1], repeat=8)))

def little_model(X):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))

    for p in range(P):
        x = X[p,None,:]
        W += np.outer(x,x)
    W[np.diag_indices(N)] = 0
    return W



def recall(data, W, steps):
    P = data.shape[0]
    for i in range(steps):
        data = np.sign(np.dot(data, W.T))
    return data.astype(int)


W = little_model(clean.copy())

ans = recall(noisy2, W, steps=901)
diffing = 0
for n, c in zip(ans, clean):
    print(n, c)
    if not ((n == c).all()):
        diffing += 1
print(diffing)

'''
obj = {}
for i in range(255):
    ans = recall(int_to_array(i), W, steps=4)
    if str(ans) in obj:
        obj[str(ans)] += 1
    else:
        obj[str(ans)] = 1

print(json.dumps(obj, indent=2, sort_keys=True), len(obj.keys()))
'''
# recall(data, W)
