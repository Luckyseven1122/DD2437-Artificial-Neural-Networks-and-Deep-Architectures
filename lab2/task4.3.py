import numpy as np
import os

from src.LoadData import LoadData
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt

districts, names, party, votes, sex = LoadData().mp()

class SOM:
    ''' SOM takes x = data, grid_size = (n,m) size of weight matrix '''

    def __init__(self, x, grid_size):
        self.x = x
        self.n_points = x.shape[0]
        self.n_weights = grid_size[0]
        self.n_features = x.shape[1]
        self.grid_size = grid_size

        self.weight_grid = self.init_weights(load_weights = True)
    
    def init_weights(self, load_weights):
        ''' initialise weight grid with numbers between 0,1, also saves the grid'''
        saved_grid_name = 'SOM_weight_grid_saved_MP'
        grid = None

        has_saved_grid = os.path.isfile(saved_grid_name)
        if(not(has_saved_grid)):
            min, max= 0, 1,
            grid = np.random.uniform(min, max, self.grid_size)
            np.savetxt(saved_grid_name, grid, fmt="%s")

        if(load_weights):
            grid = np.array(np.loadtxt(saved_grid_name))

        assert type(grid) is np.ndarray, " Failed to init grid"
        return grid

    def find_nearest_weight(self, prop):
        ''' find closets weight to a given input prop returns: (idx, distance) = (int, float)'''
        distances = np.linalg.norm(prop - self.weight_grid, axis=1)
        arg_min = distances.argmin()
        return arg_min, distances[arg_min]
    
    
    def get_neighbours(self, idx, radius = 1):
        ''' returns the weights within the square of radius = radius around weight[idx] '''

        w = np.arange(0,100).reshape((10,10))
        twoD = [(j,i) for j in range(10) for i in range(10)]
        row, col = twoD[idx]
        bros = w[max(row - radius, 0) : min(row + radius + 1,10), max(col - radius, 0) : min(col + radius + 1 ,10)].flatten()
        return bros[bros != idx]
        
   
    def train(self, epochs, eta, radius):
        radius_smoothing = 10 # how fast radius should shrink
        for epoch in range(epochs):
            for i, prop in enumerate(self.x):
                nb, dist = self.find_nearest_weight(prop)
                self.weight_grid[nb] += eta*(prop - self.weight_grid[nb]) 

                ''' get neighbours around the winning weight '''
                neighbours_idx = self.get_neighbours(idx = nb, radius = radius)
                neighbours = self.weight_grid[neighbours_idx]

                self.weight_grid[neighbours_idx] += eta * (prop - neighbours)
                
                ''' update learning rate '''
                radius = int(np.ceil(5 * np.exp(-epoch / radius_smoothing)))
    

    def result(self):
        ''' for each mp find the closest weight'''
        pos = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)
        return pos

def add_legend(d):
    d.pop('null')
    patchList = []
    for key in d:
            if(key == 'null'): continue
            data_key = mpatches.Patch(color=d[key], label=key)
            patchList.append(data_key)
    
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), prop={'size': 30})


# best settings for parties was 20 epochs, eta 0.2, radius 349,radius_smoothing=10, alpha const = 5
settings = {
    'grid_size': (10**2,31),
    'data': votes,
    'epochs': 20,
    'eta': 0.2,
    'radius': 349
}

SOM = SOM(settings['data'], settings['grid_size'])

SOM.train(settings['epochs'], settings['eta'], settings['radius'])
res  = SOM.result()

buckets = defaultdict(list)

for mp_idx, weight in enumerate(res):
    buckets[weight].append((mp_idx, party[mp_idx], sex[mp_idx], districts[mp_idx], names[mp_idx]))

party_colors = {
    'v':  "#DA291C", 
    's':  "#E8112d", 
    'mp': "#83CF39", 
    'c':  "#009933",
    'fp': "#006AB3",
    'm':  "#52BDEC", 
    'kd': "#000077",
    'null': '#FFFFFF'}
p_map = { 'v': 0,  's': 1,  'mp': 2,  'c': 3, 'fp':4,  'm': 5, 'kd':6,  'null':7}

sex_colors = {
    'Boys': '#0000ff', # boy
    'Girls': '#ffc0cb', # gurl
    'null': '#FFFFFF'
}
data_to_plot_x, data_to_plot_y = np.zeros(100), np.zeros(100)
twoD = [(j,i) for j in range(10) for i in range(10)]
xx, yy = zip(*twoD)
xx = np.array(xx)
yy = np.array(yy)
colors = [party_colors['null']]*100

boys_c, boy_size = [sex_colors['null']]*100, [200]*100
girls_c, girl_size = [sex_colors['null']]*100, [200]*100

sizes = [200]*100

menu = 2 #1 # plot parties
scale = 350
for b in range(100):
    bucket = list(zip(*buckets[b]))
    print(len(bucket))
    if len(bucket) == 0:
        xx[b] = -1
        yy[b] = -1
        continue

    mp_ids, parties, sexes, dists, n = bucket 
    if(menu == 1):
        parties, counts = np.unique(parties, return_counts=True)
        print(parties, counts)
        arg_max = np.argmax(counts)
        colors[b] = party_colors[parties[arg_max]]
        sizes[b] = counts[arg_max] * scale
        
        continue

    if(menu == 2): # plot sexes
        sex, counts = np.unique(sexes, return_counts=True)
        print(sex, counts, ' cords; ', xx[b], yy[b])
        if(len(sex) == 2):
            b_c, g_c = counts[0], counts[1]
            boy_size[b], girl_size[b] = b_c * scale, g_c * scale
            boys_c[b], girls_c[b] = sex_colors['Boys'], sex_colors['Girls']
        elif(sex[0] == 0): # man
            boy_size[b] = counts[0] * scale
            boys_c[b] = sex_colors['Boys']
            girl_size[b] = 0
        else: # gurl
            girl_size[b] = counts[0] * scale
            girls_c[b] = sex_colors['Girls']
            boy_size[b] = 0
        continue

    if(menu == 3): # plot districts
        pass # TODO



plt.style.use('seaborn-whitegrid')

if(menu == 1):
    plt.scatter(xx,yy, c=colors, s=sizes, alpha=0.9, cmap=party_colors.values())
    img = np.array(colors).reshape((10,10))
    add_legend(party_colors) 
    plt.title('Voting patterns for the Swedish parlament, 2004-2005', fontsize=36, y=1.05)

if(menu == 2):
    plt.scatter(xx,yy, c=boys_c, s=boy_size, alpha=0.8, marker="D")
    plt.scatter(xx,yy, c=girls_c, s=girl_size, alpha=0.8)
    add_legend(sex_colors)
    plt.title('MP gender cluster based on voting, 2004-2005', fontsize=36, y=1.05)

ax = plt.gca()
ax.set_aspect('equal')
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))
ax.set_xticklabels(np.arange(0, 10, 1), fontsize=18)
ax.set_yticklabels(np.arange(0, 10, 1), fontsize=18)
plt.show()

