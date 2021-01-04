import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def init_square(nu=10, nv=10):
    x_axis = np.linspace(0., 1., nu)
    y_axis = np.linspace(0., 1., nv)
    x, y = np.meshgrid(x_axis, y_axis)
    return np.array((x.flatten(), y.flatten())).T

class Tri2D(object):

    def __init__(self, points):
        self.coords = points
        self.nodes = range(points.shape[0])
        self.tri = Delaunay(points)
        self.edges = []
        for (i, j, k) in self.tri.simplices:
            if (i,j) not in self.edges:
                self.edges.append((i,j))
            if (j,i) not in self.edges:
                self.edges.append((j,i))
            if (i,k) not in self.edges:
                self.edges.append((i,k))
            if (k,i) not in self.edges:
                self.edges.append((k,i))
            if (j,k) not in self.edges:
                self.edges.append((j,k))
            if (k,j) not in self.edges:
                self.edges.append((k,j))
            #self.edges += [(i, j), (j, i)]
            #self.edges += [(i, k), (k, i)]
            #self.edges += [(j, k), (k, j)]
        print('Done')

    def plot(self):
        plt.figure()
        plt.triplot(self.coords[:, 0], self.coords[:, 1], self.tri.simplices)
        plt.show()



if __name__ == '__main__':
    points = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0]])
    tri2d = Tri2D(points)
    print(tri2d)
    tri2d.plot()
