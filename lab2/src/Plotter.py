import numpy as np
import matplotlib.pyplot as plt

# CONFIG PARAMETERS
RBF_COLOR = 'r'
RBF_FACECOLOR = "none"
RBF_ALPHA = 0.2



def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_centroids_1d(centroids, sigma):
    c = centroids.get_matrix()
    fig = plt.gcf()
    ax = fig.gca()
    x_coords = c[0]
    for x in x_coords:
        circle = plt.Circle((x, 0), sigma, color='r')
        circle.set_clip_box(ax.bbox)
        circle.set_edgecolor( RBF_COLOR )
        circle.set_facecolor( RBF_FACECOLOR )
        circle.set_alpha( RBF_ALPHA )
        ax.add_artist(circle)
