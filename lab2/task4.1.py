import numpy as np
import os
from src.LoadData import LoadData

animals, animal_names = LoadData().animals()

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
        saved_grid_name = 'SOM_weight_grid_saved'
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

        behind = max(0, idx - look_around)
        infront = min(idx + look_around, len(weight_idxes))

        neighbours = weight_idxes[behind: infront]
        return neighbours[neighbours != idx]
        
   
    def train(self, epochs, eta, radius):
        radius_smoothing = 2 # how fast radius should shrink
        for epoch in range(epochs):
            for i, prop in enumerate(self.x):
                nb, dist = self.find_nearest_weight(prop)
                self.weight_grid[nb] += eta*(prop - self.weight_grid[nb]) 

                ''' get neighbours around the winning weight '''
                neighbours_idx = self.get_neighbours(idx = nb, radius = radius)
                neighbours = self.weight_grid[neighbours_idx]

                self.weight_grid[neighbours_idx] += eta * (prop - neighbours)
                
                ''' update learning rate '''
                radius = int(np.ceil(epochs * np.exp(-epoch / radius_smoothing)))
    
    ''' for each animal find the closest weight'''
    def result(self, animal_names):
        pos = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)

        # sort by argsort here is yolo but only stuff that makes sense......
        return animal_names[np.argsort(pos)].reshape(32)

settings = {
    'grid_size': (100,84),
    'data': animals,
    'epochs': 20,
    'eta': 0.2,
    'radius': 50
}

SOM = SOM(settings['data'], settings['grid_size'])

SOM.train(settings['epochs'], settings['eta'], settings['radius'])
''' now similar animals should be group together '''

print(SOM.result(animal_names))
