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
    return - np.einsum('ij,i,j', W, x, x)



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
    return W / N

def save_snapshot(X, iter, energy):
    plt.clf()
    gca = plt.gca()
    gca.axes.get_yaxis().set_visible(False)
    gca.axes.get_xaxis().set_visible(False)
    plt.suptitle('iteration: ' +str(iter))
    plt.title('E = {:7.2f}'.format(energy))
    plt.imshow(X.reshape(32,32))
    plt.savefig('./figures/' + str(iter) + '.png')

def recall_sequential(X, W, steps, random = False):
    X, W = X.copy(), W.copy()
    E = 0
    counter = 0
    # save_snapshot(X, counter, E)
    eng_saved = []
    for _ in range(steps):
        for p in range(X.shape[0]):
            # E = energy(X[0], W)
            # eng_saved.append(E)
            # print('energy: ', E)
            for i in range(X.shape[1]):
                a_i = 0
                idx = np.random.randint(X.shape[1]) if random else i
                for j in range(X.shape[1]):
                    if(idx == j):
                        continue
                    a_i += W[idx][j] * X[p][j]


                X[p][idx] = np.sign(a_i)
                counter += 1
                # if (counter % 1) == 0:
                E = energy(X[0], W)
                eng_saved.append(E)
                if(random):
                    print('energy: ', E)
                # pass
                    # save_snapshot(X, counter, energy(X[0], W))

    return X.astype(int), eng_saved


def task3_3_plot_energy:

    W = little_model(pics[0:3,:])
    steps = 10
    ans_rand, eng_rand = recall_sequential(pics[9,None,:], W, steps=steps, random=True)
    ans_seq, eng_seq = recall_sequential(pics[9,None,:], W, steps=steps, random=False)

    x = np.arange(len(eng_rand))
    
    font = {'family' : 'normal',
            'size'   : 18}
    
    plt.rc('font', **font)
    plt.title('Sequential Recall Energy')
    plt.plot(x, eng_rand, label='Random index')
    plt.plot(x, eng_seq, label='Sequential index')
    plt.legend()
    plt.show()


# W = little_model(pics[0:3,:])
# steps = 10
# ans_rand, eng_rand = recall_sequential(pics[9,None,:], W, steps=steps, random=True)
# ans_seq, eng_seq = recall_sequential(pics[9,None,:], W, steps=steps, random=False)

# show_pics(pics, 'all')
# show_pics(ans, 'Recall: 0-3')
