import numpy as np
import matplotlib.pyplot as plt
import json

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
    W = np.random.normal(0, 2, W.shape)
    #W = 0.5*(W+W.T)
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
    #save_snapshot(X, counter, E)
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

                E = energy(X[0], W)
                eng_saved.append(E)
                if (counter % 100) == 0:
                    pass
                    #save_snapshot(X, counter, energy(X[0], W))


    return X.astype(int), eng_saved

def task3_3_plot_energy():
    W = little_model(pics[0:3,:])
    steps = 10
    ans_rand, eng_rand = recall_sequential(pics[9,None,:], W, steps=steps, random=True)
    ans_seq, eng_seq = recall_sequential(pics[9,None,:], W, steps=steps, random=False)

    x = np.arange(len(eng_rand))
     
    plt.subplot(121)
    plt.ylabel('Energy')
    plt.xlabel('iteration')

    plt.title('Sequential and Random sampling recall')
    plt.plot(x, eng_rand, label='Random sampling')
    plt.plot(x, eng_seq, label='Sequential sampling')

    plt.ylabel('Energy')
    plt.xlabel('iteration')

    plt.legend()

    plt.subplot(122)

    W = np.random.normal(0, 1, W.shape)
    Wsym = 0.5 * (W * W.T)
    print('asym')
    _,ans = recall_sequential(pics[9,None,:], W, steps=10)
    print('sym')
    _,an = recall_sequential(pics[9,None,:], Wsym, steps=10)
    
    plt.plot(np.arange(0, len(ans)), ans, label="Asymmetric W")
    plt.plot(np.arange(0, len(an )), an , label="Symmetric W")
    plt.title('Energy on W with normal distribution')
    plt.ylabel('Energy')
    plt.xlabel('iteration')

    plt.legend()
    plt.show()


task3_3_plot_energy()
# W = little_model(pics[0:3,:])
# steps = 10
# ans_rand, eng_rand = recall_sequential(pics[9,None,:], W, steps=steps, random=True)
# ans_seq, eng_seq = recall_sequential(pics[9,None,:], W, steps=steps, random=False)

W = little_model(pics[0:3,:])
ans = recall_sequential(pics[0,None,:], W, steps=10)

# show_pics(pics, 'all')
# show_pics(ans, 'Recall: 0-3')
