import numpy as np
import os

from src.LoadData import LoadData
from task43 import SOM

from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt

districts, names, party, votes, sex = LoadData().mp()

party_colors = {
    'v':  "#DA291C", 
    's':  "#E8112d", 
    'mp': "#83CF39", 
    'c':  "#009933",
    'fp': "#006AB3",
    'm':  "#52BDEC", 
    'kd': "#000077",
    'no party': '#000000'}

p_map = { 'v': 0,  's': 1,  'mp': 2,  'c': 3, 'fp':4,  'm': 5, 'kd':6, 'no party': 7}

sex_colors = {
    'Boys': '#0000ff', # boy
    'Girls': '#ffc0cb', # gurl
    'null': '#FFFFFF'
}


def add_legend(d):
    d.pop('null')
    patchList = []
    for key in d:
            if(key == 'null'): continue
            data_key = mpatches.Patch(color=d[key], label=key)
            patchList.append(data_key)
    
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), prop={'size': 30})


def run_SOM():
    # best settings for parties was 20 epochs, eta 0.2, radius 349,radius_smoothing=10, alpha const = 5
    settings = {
        'grid_size': (10**2,31),
        'data': votes,
        'epochs': 20,
        'eta': 0.2,
        'radius': 349
    }
    
    som = SOM(settings['data'], settings['grid_size'])
    
    som.train(settings['epochs'], settings['eta'], settings['radius'])
    res  = som.result()
    return res


def create_buckets():
    buckets = defaultdict(list)
    for mp_idx, weight in enumerate(res):
        buckets[weight].append((mp_idx, party[mp_idx], sex[mp_idx], districts[mp_idx], names[mp_idx]))
    return buckets

''' run SOM and place each MP in the grid '''
res = run_SOM()
buckets = create_buckets()

''' coordinates for 10x10 grid '''
twoD = [(j,i) for j in range(10) for i in range(10)]
xx, yy = zip(*twoD)
xx, yy = np.array(xx), np.array(yy)

''' party colors '''
# colors = [party_colors['null']]*100
boys_c, boy_size = [sex_colors['null']]*100, [200]*100
girls_c, girl_size = [sex_colors['null']]*100, [200]*100


''' plot util scaling and sizing '''
sizes = [200]*100
scale = 350


party_buffer = np.zeros((100, 8))
sex_buffer = np.zeros((100, 2))
dist_buffer = np.zeros((100, 29))
img = np.zeros((10,10))

menu = 3 # for districts #2 for sex #1 # plot parties

for b in range(100):
    bucket = list(zip(*buckets[b]))
    if len(bucket) == 0:
        # xx[b] = -1
        # yy[b] = -1
        continue

    mp_ids, parties, sexes, dists, n = bucket 

    if(menu == 1):
        p, counts = np.unique(parties, return_counts=True)
        arg_max = np.argmax(counts)
        
        for party_name, c in zip(p, counts):
            print(b, p_map[party_name], party_name)
            party_buffer[b][p_map[party_name]] = c
        print(p, counts)

        placeholder = [0]*7 # hold counts for each party
        # colors[b] = party_colors[p[arg_max]]
        sizes[b] = counts[arg_max] * scale
        continue

    if(menu == 2): # plot sexes
        sex, counts = np.unique(sexes, return_counts=True)
        print(sex, counts)
        if(len(sex) == 2):
            b_c, g_c = counts[0], counts[1]
            boy_size[b], girl_size[b] = b_c * scale, g_c * scale

            sex_buffer[b][0] = b_c
            sex_buffer[b][1] = g_c

            boys_c[b], girls_c[b] = sex_colors['Boys'], sex_colors['Girls']

            img[xx[b]][yy[b]] = b_c
        elif(sex[0] == 0): # man
            boy_size[b] = counts[0] * scale
            boys_c[b] = sex_colors['Boys']
            girl_size[b] = 0

            sex_buffer[b][0] = counts[0]
            img[xx[b]][yy[b]] = counts[0]
        else: # gurl
            girl_size[b] = counts[0] * scale
            girls_c[b] = sex_colors['Girls']
            boy_size[b] = 0

            sex_buffer[b][1] = counts[0]
        continue

    if(menu == 3): # plot districts
        d, d_counts = np.unique(dists, return_counts=True)
        for d_d, d_c in zip(d,d_counts):
            print(d_d, d_c)
            dist_buffer[b][d_d - 1] = d_c # districts are 1 indexed......
        

print(dist_buffer)
# plt.style.use('seaborn-whitegrid')

# fig, ax = plt.subplots()
# ax = plt.gca()

if(menu == 1):
    for i in range(8):
        print(list(p_map.keys()))
        p_c = party_colors[list(p_map.keys())[i]]
        party_data = party_buffer[:,i]

        xy = np.column_stack([party_data, np.arange(0,100)])
        plt.scatter(xx,yy, c=p_c, s=300 * i, alpha=1, marker=xy)

    # img = np.array(colors).reshape((10,10))
    # add_legend(party_colors) 
    # plt.title('Voting patterns for the Swedish parlament, 2004-2005', fontsize=36, y=1.05)

if(menu == 2):
    print(img)
    # plt.scatter(xx,yy, c=boys_c, s=200, alpha=0.8, marker=sex_buffer)
    plt.imshow(img, extent=[0,0, 10, 10], clip_on=True)
    # plt.scatter(xx,yy, c=boys_c, s=boy_size, alpha=0.8, marker="D")
    # plt.scatter(xx,yy, c=girls_c, s=girl_size, alpha=0.8)
    add_legend(sex_colors)
    plt.title('MP gender cluster based on voting, 2004-2005', fontsize=36, y=1.05)

if(menu == 3):
    fig, axes = plt.subplots(10, 10)
    for i, ax in enumerate(axes.flatten()):
        norm = np.sum(dist_buffer[i])
        if(norm == 0):
            ax.axis('off')
            continue
        x = dist_buffer[i] / norm if norm > 0 else 1 # normalize the row so all sums to 1 
        # print(x)
        ax.pie(x, radius = 1, autopct="", pctdistance=1)


ax.set_aspect('equal')
plt.axis('equal')
# ax.set_xticks(np.arange(0, 10, 1))
# ax.set_yticks(np.arange(0, 10, 1))
# ax.set_xticklabels(np.arange(0, 10, 1), fontsize=18)
# ax.set_yticklabels(np.arange(0, 10, 1), fontsize=18)
plt.show()

