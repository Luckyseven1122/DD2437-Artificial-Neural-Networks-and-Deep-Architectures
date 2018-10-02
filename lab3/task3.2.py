import numpy as np
import matplotlib.pyplot as plt
import json

pics = None
with open('pict.dat') as f:
    pics = np.fromfile(f, sep=',')
    pics = pics.reshape(11, 1024)

def show_pics(pics):
    n_pics = pics.shape[0]
    fig, ax = plt.subplots(1, n_pics)
    for i in range(n_pics):
        pic = pics[i, :].reshape(32, 32)
        plt.subplot(1, n_pics, i+1)
        gca = fig.gca()
        gca.axes.get_yaxis().set_visible(False)
        gca.axes.get_xaxis().set_visible(False)
        plt.imshow(pic)
    plt.show()

#show_pics(pics) # all pics
#show_pics(pics[0,:]) # for one pic
#show_pics(pics[0:4,:]) # a set of pics

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
