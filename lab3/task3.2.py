import numpy as np


pics = None

with open('pict.dat') as f:
    pics = np.fromfile(f, sep=',')
    pics = pics.reshape(11, 32, 32)
