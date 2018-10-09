import numpy as np
import matplotlib.pyplot as plt
import json
import math


def get_sparse_pattern(activity_percentage, N):
    n_active = math.floor(activity_percentage * N)
    sample = np.zeros((1, N))
    indices = np.arange(0, N)
    np.random.shuffle(indices)
    for i in range(n_active):
        sample[0,indices[i]] = 1
    return sample


def get_pattern_batch(n_samples, activity_percentage):
    N = 300
    patterns = np.zeros((n_samples, N))
    for i in range(n_samples):
        patterns[i,:] = get_sparse_pattern(activity_percentage, N)
    return patterns


def energy(x, W):
    return - np.einsum('ij,i,j', W, x, x)


def network(X, p):
    P = X.shape[0] # P patterns
    N = X.shape[1] # N Units
    W = np.zeros((N,N))
    #print('Number of patterns:', P)
    #print('Number of units:', N)
    for i in range(P):
        x = X[i,None,:]
        W += np.outer(x - p, x - p)
    #W[np.diag_indices(N)] = 0
    return W


def recall_sequential(X, W, bias, steps, random=False):
    X, W = X.copy(), W.copy()
    for _ in range(steps):
        for p in range(X.shape[0]):
            for i in range(X.shape[1]):
                a_i = 0
                idx = np.random.randint(X.shape[1]) if random else i
                for j in range(X.shape[1]):
                    if(idx == j):
                        continue
                    a_i += W[idx][j] * X[p][j]
                X[p][idx] = 0.5 + 0.5*np.sign(a_i - bias)
    return X.astype(int)


def similarity(o, a):
    same = 0
    N = o.shape[1]
    for org, ans in zip(o.T, a.T):
        if(org == ans):
            same += 1
    return same, True if same == N else False




activity_percentages = [0.1]
bias = [0.5, 1, 1.5, 2]
offset = [-0.1, -0.025, 0.025, 0.1]
colors = ['b', 'g', 'r']
batch_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
number_of_iterations = 20
for p in activity_percentages:
    data = {}
    for b in bias:
        data[str(b)+'_mean'] = []
        data[str(b)+'_std'] = []
        counter = 0
        for n_samples in batch_sizes:
            print(counter)
            recall_set = []
            for i in range(number_of_iterations):
                batch = get_pattern_batch(n_samples, p)
                W = network(batch, p)

                # recall phase
                recall = 0
                for i in range(n_samples):
                    sample = batch[i,None,:]
                    ans = recall_sequential(X=sample, W=W, bias=b, steps=5, random=True)
                    correct, good = similarity(sample, ans)
                    if good:
                        recall += 1
                recall_set.append(recall)
            counter += 1
            data[str(b)+'_mean'].append(np.mean(recall_set))
            data[str(b)+'_std'].append(np.std(recall_set))
    for i, b in enumerate(bias):
        plt.errorbar(np.array(batch_sizes) + offset[i], data[str(b)+'_mean'], data[str(b)+'_std'],
                        label='bias='+str(b),
                        marker='_',
                        capsize=5)
    plt.legend()
    plt.title('Sparse Patterns (1% Activity)')
    plt.ylabel('Correctly Recalled Patterns')
    plt.xticks(np.array(batch_sizes))
    plt.xlabel('Number of Patterns')
    plt.show()
