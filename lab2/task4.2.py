import numpy as np
import os
import matplotlib.pyplot as plt
from src.LoadData import LoadData

cities = LoadData().cities()

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
        saved_grid_name = 'SOM_weight_grid_saved_CITIES'
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
        ''' returns the weights behind and infront of weight[idx] '''

        weight_idxes = np.arange(self.n_weights)

        look_around = int(np.floor(radius / 2))
        
        ''' if lookaround is to small we have no neighbours '''
        if(look_around < 1):
            return []

        ''' first and last node needs to be connected so we need to wrap around and connect them.. '''
        bros_behind, bros_infront = [], []
        if(idx == 0):
            return [weight_idxes[9], weight_idxes[1]]
        
        if(idx == 9):
            return [weight_idxes[0], weight_idxes[8]]

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
    

    def result(self, cities):
        ''' for each animal find the closest weight'''
        pos = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)
            # print(nb)

        # sort by argsort here is yolo but only stuff that makes sense......
        # return cities[np.argsort(pos)]
        return cities[pos]

settings = {
    'grid_size': (10,2),
    'data': cities,
    'epochs': 20,
    'eta': 0.2,
    'radius': 2
}

SOM = SOM(settings['data'], settings['grid_size'])

SOM.train(settings['epochs'], settings['eta'], settings['radius'])

tour = SOM.weight_grid
x = tour[:,0]
y = tour[:,1]

con_x = [x[0], x[-1]]
con_y = [y[0], y[-1]]

plt.scatter(x[0], y[0], label = 'Start', s=800, facecolors='none', edgecolors='r')
plt.scatter(x[-1], y[-1], label = 'End', s=800, facecolors='none', edgecolors='b')
plt.plot(x,y, 'r', label = 'Suggested tour')
plt.plot(con_x, con_y, 'r') # connect last and first city lol
plt.scatter(x,y, s=100, c='b', label = 'Approximated cities')
plt.scatter(cities[:,0], cities[:,1], marker=('x'), s=150, label='Actual cities')
plt.legend(prop={'size': 20})
plt.xlabel('X coordinates', fontsize=18)
plt.ylabel('Y coordinates', fontsize=18)

plt.show(block=True)
