import numpy as np
import matplotlib.pyplot as plt




def generate_classes():
    n_points = 100
    mA = np.array([ 1.0, 0.5])
    mB = np.array([-1.2, 0.5])
    sigmaA = 0.5
    sigmaB = 0.5

    classA = np.zeros([2, n_points])
    classB = np.zeros([2, n_points])
    classA[0,:] = np.random.randn(1, n_points) * sigmaA + mA[0]
    classA[1,:] = np.random.randn(1, n_points) * sigmaA + mA[1]
    classB[0,:] = np.random.randn(1, n_points) * sigmaB + mB[0]
    classB[1,:] = np.random.randn(1, n_points) * sigmaB + mB[1]
    return classA, classB


def plot_classes(classes):
    for idx, c in enumerate(classes):
        color = 'C' + str(idx) + 'o'
        plt.plot(c[0,:], c[1,:], color)
    plt.show()


classA, classB = generate_classes()


plot_classes([classA, classB])
