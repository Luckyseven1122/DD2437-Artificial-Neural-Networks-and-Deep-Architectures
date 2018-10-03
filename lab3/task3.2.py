import numpy as np
import matplotlib.pyplot as plt
import json

pics = None
with open('pict.dat') as f:
    pics = np.fromfile(f, sep=',')
    pics = pics.reshape(11, 1024)

def show_pics(pics, title):
    n_pics = pics.shape[0]
    fig, ax = plt.subplots(1, n_pics)
    plt.suptitle(title)
    for i in range(n_pics):
        pic = pics[i, :].reshape(32, 32)
        plt.subplot(1, n_pics, i+1)
        gca = fig.gca()
        gca.axes.get_yaxis().set_visible(False)
        gca.axes.get_xaxis().set_visible(False)
        plt.imshow(pic)

    plt.show()

#show_pics(pics, 'all') # all pics
#show_pics(pics[0,:], 'first') # for one pic
#show_pics(pics[0:4,:], '0-4') # a set of pics

def energy(x, W):
    return - np.einsum('ij,i,j', W, x, x) / 2



def little_model(X):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))
    print('Number of patterns:', P)
    print('Number of units:', N)
    for p in range(P):
        x = X[p,None,:]
        W += np.outer(x,x)
    W[np.diag_indices(N)] = 0
    return W



def recall_sequential(X, W, steps):
    E = 0
    for _ in range(steps):
        for p in range(X.shape[0]):
            for i in range(X.shape[1]):
                a_i = 0
                for j in range(X.shape[1]):
                    if(i == j):
                        continue
                    a_i += W[i][j] * X[p][j]
                x_new = np.sign(a_i)
                x_old = X[p][i]
                s = a_i
                E = energy(X[0], W)
                print('energy: ', E)
                X[p][i] = np.sign(a_i)
    return X.astype(int)




W = little_model(pics[0:3,:])
ans = recall_sequential(pics[10,None,:], W, steps=3)

show_pics(pics[0:3,:], 'Before 0-3')
show_pics(ans, 'Recall: 0-3')
