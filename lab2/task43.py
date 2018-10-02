import numpy as np
import os

from src.LoadData import LoadData
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt

# districts, names, party, votes, sex = LoadData().mp()

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


