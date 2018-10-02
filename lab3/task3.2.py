import numpy as np
import matplotlib.pyplot as plt
import json

pics = None
with open('pict.dat') as f:
    pics = np.fromfile(f, sep=',')
    pics = pics.reshape(11, 32, 32)

def show_pics():
    fig, ax = plt.subplots(1, 11)
    for i in range(11):
        plt.subplot(1, 11, i+1)
        gca = fig.gca()
        gca.axes.get_yaxis().set_visible(False)
        gca.axes.get_xaxis().set_visible(False)
        plt.imshow(pics[i])
    plt.show()

show_pics()
