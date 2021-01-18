import numpy as np


class DiGraph3D(object):

    def __init__(self, nu=10, nv=10, nw=10):
        self.nu = nu
        self.nv = nv
        self.nw = nw
        self.coords = None
        self.nodes = None
        self.edges = None
        self.__build()

    def _get_index(self, iu, iv, iw):
        return iu + iv * self.nu + iw * self.nu * self.nv

    def _get_uvw(self, node):
        iw = node // (self.nu * self.nv)
        iv = node // self.nu - iw * self.nv
        iu = node - iv * self.nu - iw * self.nu * self.nv
        return iu, iv, iw

    def _get_neighbours(self, node, dw=1):
        neighbours = []
        iu, iv, iw = self._get_uvw(node)
        iw_n = iw + dw
        if 0 <= iw_n <= self.nw - 1:
            for iu_n in range(iu - 1, iu + 2):
                if 0 <= iu_n <= self.nu - 1:
                    for iv_n in range(iv - 1, iv + 2):
                        if 0 <= iv_n <= self.nv - 1:
                            neighbours.append(self._get_index(iu_n, iv_n, iw_n))
        return neighbours

    def get_in_edges(self, node):
        return [(neighbor, node) for neighbor in self._get_neighbours(node, dw=-1)]

    def get_out_edges(self, node):
        return [(node, neighbor) for neighbor in self._get_neighbours(node, dw=1)]

    def get_edges(self, node):
        return self.get_in_edges(node), self.get_out_edges(node)

    def __build(self):
        x_axis = np.linspace(0., 1., self.nu)
        y_axis = np.linspace(0., 1., self.nv)
        z_axis = np.linspace(0., 1., self.nw)
        n_nodes = self.nu * self.nv * self.nw
        self.nodes = range(n_nodes)
        self.coords = np.zeros((n_nodes, 3))
        self.edges = []
        for iw in range(self.nw):
            for iv in range(self.nv):
                for iu in range(self.nu):
                    node = self._get_index(iu, iv, iw)
                    self.coords[node, 0] = x_axis[iu]
                    self.coords[node, 1] = y_axis[iv]
                    self.coords[node, 2] = z_axis[iw]
                    self.edges += self.get_out_edges(node)

    def _get_crossing_edges(self, edge):
        (n1, n2) = edge
        iu1, iv1, iw1 = self._get_uvw(n1)
        iu2, iv2, iw2 = self._get_uvw(n2)
        crossing = [(self._get_index(iu2, iv2, iw1),
                     self._get_index(iu1, iv1, iw2))]
        if iv2 != iv1 and iu2 != iu1:
            crossing.append((self._get_index(iu2, iv1, iw1),
                             self._get_index(iu1, iv2, iw2)))
            crossing.append((self._get_index(iu1, iv2, iw1),
                             self._get_index(iu2, iv1, iw2)))
        return crossing

    def get_all_crossing_edges(self):
        crossing = []
        for iw in range(self.nw - 1):
            for iv in range(self.nv - 1):
                for iu in range(self.nu - 1):
                    e1 = (self._get_index(iu, iv, iw),
                          self._get_index(iu + 1, iv, iw + 1))
                    crossing.append(self._get_crossing_edges(e1) + [e1])
                    e2 = (self._get_index(iu, iv, iw),
                          self._get_index(iu + 1, iv + 1, iw + 1))
                    crossing.append(self._get_crossing_edges(e2) + [e2])
                    e3 = (self._get_index(iu, iv, iw),
                          self._get_index(iu, iv + 1, iw + 1))
                    crossing.append(self._get_crossing_edges(e3) + [e3])
        return crossing


if __name__ == "__main__":
    dg3d = DiGraph3D()
    n_nodes = len(dg3d.nodes)
    n_edges = len(dg3d.edges)
    print(f'Digraph 3d: {n_nodes} nodes, {n_edges} edges')
    print(dg3d.get_out_edges(12))
    crossing_edges = dg3d.get_all_crossing_edges()
    ce_list = crossing_edges[142]
    print(ce_list)
    for edge in ce_list:
        c = 0.5 * (dg3d.coords[edge[0]] + dg3d.coords[edge[1]])
        print(c)
