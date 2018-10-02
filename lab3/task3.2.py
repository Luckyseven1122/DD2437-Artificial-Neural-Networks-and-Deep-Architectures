import numpy as np
with open('pict.dat') as f:
    lines = np.array([int(p.strip()) for p in f.readlines()[0].split(',')])
    print(lines.shape)
