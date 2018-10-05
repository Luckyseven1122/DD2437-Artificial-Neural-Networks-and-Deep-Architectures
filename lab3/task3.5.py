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
    eng_saved = []
    for _ in range(steps):
        for p in range(X.shape[0]):
            E = energy(X[0], W)
            eng_saved.append(E)
            print('energy: ', E)
            for i in range(X.shape[1]):
                a_i = 0
                idx = np.random.randint(X.shape[1]) if random else i
                for j in range(X.shape[1]):
                    if(idx == j):
                        continue
                    a_i += W[idx][j] * X[p][j]
                X[p][idx] = np.sign(a_i)
                counter += 1

                # E = energy(X[0], W)
                # eng_saved.append(E)
                if (counter % 100) == 0:
                    #save_snapshot(X, counter, energy(X[0], W))
                    pass


    return X.astype(int), eng_saved

def correct_picture(org_pic, ans_pic):
    diff = 0
    for org, ans in zip(org_pic,ans_pic):
        if(org != ans):
            diff += 1
    return diff, True if diff == 0 else False

def noise_and_recall(model, p):
    W = model
    noise_amount = 10
    noise_idx = np.arange(0, 1024)
    np.random.shuffle(noise_idx)
    
    arr = np.arange(0, float(noise_amount/100)*1024).astype(int).tolist()
    pic = p.copy()
    for j in arr:
        pic[0, noise_idx[j]] = 1 if pic[0, noise_idx[j]] == -1 else -1
    
    pic = pic.astype(int)
    ans, eng = recall_sequential(pic, W, steps=10, random=True)
    return ans, eng

def get_random_pattern():
    return np.array([1 if p > 0.5 else -1 for p in np.random.rand(1024)]).reshape((1,1024))

def test_memory(patterns, p):
    diff_buffer = []
    memory_pattern = []
    patterns_in_memory = []
    for i in range(30):
        memory_pattern.append(patterns[i])
        print('Training model with patterns: ', str(len(memory_pattern)))
        mem = ['p' + str(j) for j in range(0,i+1)]
        mem = ','.join([j + '\n' if j == 'p4' else j for j in mem])
        print(mem)
        patterns_in_memory.append(mem)
        print(patterns_in_memory)
        W = little_model(np.vstack(memory_pattern))
    
        print('Recalling img')
   
        print(p0)
        ans, eng = noise_and_recall(W,p0)
        diffs, correct = correct_picture(p[0], ans[0])
        diff_buffer.append(diffs)
        print('Picture was restored: ', correct, ' diffed : ', diffs)
    return diff_buffer, patterns_in_memory


p0, p1, p2, p3, p4, p5, p6, p7, p8 = pics[0,None,:],pics[1,None,:],pics[2,None,:],pics[3,None,:],pics[4,None,:],pics[5,None,:],pics[6,None,:],pics[7,None,:],pics[8,None,:]

patterns = [p0, p1, p2,p3,p4,p5,p6,p7,p8]
random_patterns = [get_random_pattern() for p in range(30)]

# p_buffer,x_label = test_memory(patterns, p0)
r_buffer,x_label = test_memory(random_patterns, get_random_pattern())

ax = plt.subplot(111)
w = 0.3
x = np.arange(len(r_buffer))
# ax.bar( x - w, p_buffer, width=w, align='center', label='Real patterns')
ax.bar( x, r_buffer,width=w,  align='center', label='Random patterns')


# plt.bar(np.arange(len(p_buffer)), p_buffer, label='Real patterns')
# plt.bar(np.arange(len(p_buffer)), r_buffer, label='Random patterns')
ax.autoscale(tight=True)
plt.title('# Errors per added pattern in memory and recalling pattern p0 (in blue) and recalling a random pattern (in orange)')
plt.ylabel('#Errors')
plt.xlabel('Pattern in memory p0 - p8')
plt.xticks([r + w for r in range(len(x))], x_label, rotation=-60)
plt.legend()
plt.show()





















# show_pics(pics, 'all')
# show_pics(np.array([ans, pic]), 'Recall: 0-3')

# W = little_model(np.vstack([p0,p1,p8,p7,p6]))
# ans = noise_and_recall(W,p0)

# diffs, correct = correct_picture(p0[0], ans[0])
# print('Picture was restored: ', correct, ' diffed : ', diffs)




# W = little_model(np.vstack([p0,p1,p8,p7,p6]))
# ans = noise_and_recall(W,p0)

# diffs, correct = correct_picture(p0[0], ans[0])
# print('Picture was restored: ', correct, ' diffed : ', diffs)




