import numpy as np


class Grid2D(object):

    def __init__(self,nu=10,nv=10):
        self.nu = nu
        self.nv = nv
        self.coords = None
        self.nodes = None
        self.edges = None
        self.__init_grid()

    def __init_grid(self):
        x_axis = np.linspace(0.,1.,self.nu)
        y_axis = np.linspace(0.,1.,self.nv)
        x,y = np.meshgrid(x_axis,y_axis)
        self.nodes = range(self.nu*self.nv)
        self.coords = np.array((x.flatten(),y.flatten())).T
        self.edges = []
        for iv in range(self.nv):
            for iu in range(self.nu):
                p = iu+iv*self.nu
                if iu>0:
                    self.edges.append((p, p-1))
                if iu<self.nu-1:
                    self.edges.append((p, p+1))
                if iv>0:
                    q = iu+(iv-1)*self.nu
                    self.edges.append((p,q))
                if iv<self.nv-1:
                    q = iu+(iv+1)*self.nu
                    self.edges.append((p,q))


if __name__=='__main__':
    grid2d = Grid2D()
    print(grid2d.nodes)
    print(grid2d.edges[:5])
        