import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from src.LoadData import LoadData

districts, names, party, votes = LoadData().mp()

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

        # has_saved_grid = os.path.isfile(saved_grid_name)
        # if(not(has_saved_grid)):
        min, max= 0, 1,
        grid = np.random.uniform(min, max, self.grid_size)
        return grid
        # np.savetxt(saved_grid_name, grid, fmt="%s")

#         if(load_weights):
#             grid = np.array(np.loadtxt(saved_grid_name))

#         assert type(grid) is np.ndarray, " Failed to init grid"
#         return grid

    def find_nearest_weight(self, prop):
        ''' find closets weight to a given input prop returns: (idx, distance) = (int, float)'''
        distances = np.linalg.norm(prop - self.weight_grid, axis=1)
        arg_min = distances.argmin()
        return arg_min, distances[arg_min]
    
    
    def get_neighbours(self, idx, radius = 1):
        ''' returns the weights behind and infront of weight[idx] '''

        weight_idxes = np.arange(self.n_weights)

        look_around = int(np.floor(radius / 2))
        
        behind = max(0, idx - look_around)
        infront = min(idx + look_around, len(weight_idxes))
        
        neighbours = weight_idxes[behind: infront]
        return neighbours[neighbours != idx]
        
   
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
                radius = int(np.ceil(2 * np.exp(-epoch / radius_smoothing)))
    

    def result(self, mp):
        ''' for each animal find the closest weight'''
        pos = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)

        return pos, np.argsort(pos)
        # return mp[pos]

settings = {
    'grid_size': (10**2,31),
    'data': votes,
    'epochs': 20,
    'eta': 0.2,
    'radius': 50
}

SOM = SOM(settings['data'], settings['grid_size'])

SOM.train(settings['epochs'], settings['eta'], settings['radius'])
indexes, res  = SOM.result(votes)

idx = np.arange(1,100)
buckets = defaultdict(list)

for i in range(349):
    buckets[indexes[i]].append((res[i], party[res[i]]))

for i in range(349):
    print(i , ' : ' , buckets[i])

# print(res)
# print(indexes)
# print(party)
# buckets = dict(idx

# print(res)
# print([party[r] for r in res])
# tour = SOM.weight_grid
# x = tour[:,0]
# y = tour[:,1]

# con_x = [x[0], x[-1]]
# con_y = [y[0], y[-1]]

# plt.scatter(x[0], y[0], label = 'Start', s=800, facecolors='none', edgecolors='r')
# plt.scatter(x[-1], y[-1], label = 'End', s=800, facecolors='none', edgecolors='b')
# plt.plot(x,y, 'r', label = 'Suggested tour')
# plt.plot(con_x, con_y, 'r') # connect last and first city lol
# plt.scatter(x,y, s=100, c='b', label = 'Approximated cities')
# plt.scatter(cities[:,0], cities[:,1], marker=('x'), s=150, label='Actual cities')
# plt.legend(prop={'size': 20})
# plt.xlabel('X coordinates', fontsize=18)
# plt.ylabel('Y coordinates', fontsize=18)

plt.show(block=True)
