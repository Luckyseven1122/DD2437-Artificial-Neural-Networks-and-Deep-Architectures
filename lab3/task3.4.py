import numpy as np
import matplotlib.pyplot as plt


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

def energy(x, W):
    return - np.einsum('ij,i,j', W, x, x)

def little_model(data):

    P = data.shape[0] # P patterns
    N = data.shape[1] # N Units
    W = np.zeros((N,N))
    for i in range(P):
        x = data[i,:].reshape(1,-1)
        W += np.outer(x.T, x)
    return W

def recall(X, W, steps):
    for i in range(steps):
        E = energy(X[0], W)
        #print(E)
        X = np.sign(np.dot(W.T, X.T).reshape(1,-1))
    return X

def save_snapshot(O, X, iter):
    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots(1,2)
    plt.suptitle('Introducing noise')

    plt.subplot(1, 2, 1)
    gca = plt.gca()
    gca.axes.get_yaxis().set_visible(False)
    gca.axes.get_xaxis().set_visible(False)
    plt.imshow(O.reshape(32,32))

    plt.subplot(1, 2, 2)
    gca = plt.gca()
    gca.axes.get_yaxis().set_visible(False)
    gca.axes.get_xaxis().set_visible(False)
    plt.imshow(X.reshape(32,32))

    plt.suptitle('Noise: ' +str(iter) +'‰')
    plt.savefig('./figures/' + str(iter) +  '.png')

W = little_model(pics[0:3,:])
noise_idx = np.arange(0, 1024)
np.random.shuffle(noise_idx)
for i in range(101):
    pic = pics[1,None,:].copy()
    arr = np.arange(0, float(i/100)*1024).astype(int).tolist()
    for j in arr:
        pic[0, noise_idx[j]] = 1 if pic[0, noise_idx[j]] == -1 else -1
    pic9 = recall(pic, W, 4)
    save_snapshot(pic, pic9, i)

show_pics(pics[0:3,:], 'Before 0-3')
show_pics(ans, 'Recall: 0-3')
