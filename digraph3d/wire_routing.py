from digraph3d import DiGraph3D
import pulp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def export_path_to_vtk(points, path, output_file):
    import vtk
    nodes = []
    for (i, j) in path:
        nodes.append(i)
        nodes.append(j)
    path_xyz = sorted([points[n, :] for n in np.unique(nodes)], key=lambda elt: elt[2])
    n_pts = len(path_xyz)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(n_pts)
    for i, pt_xyz in enumerate(path_xyz):
        vtk_points.SetPoint(i, *pt_xyz)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n_pts)
    for i in range(n_pts):
        lines.InsertCellPoint(i)
    polygon = vtk.vtkPolyData()
    polygon.SetPoints(vtk_points)
    polygon.SetLines(lines)
    # write vtk file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polygon)
    writer.Write()


# display available solvers
solver_list = pulp.listSolvers(onlyAvailable=True)
print(solver_list)

# build 3d graph
nu, nv, nw = 20, 20, 20
g = DiGraph3D(nu, nv, nw)

# wires start and stop
source_uvw = [(0, 0, 0),
              (1, 0, 0),
              (1, 1, 0),
              (0, 1, 0)]

target_uvw = [(nu - 1, nv - 1, nw - 1),
              (0, 0, nw - 1),
              (1, 0, nw - 1),
              (1, 1, nw - 1)]

source_nodes = [g._get_index(*uvw) for uvw in source_uvw]
target_nodes = [g._get_index(*uvw) for uvw in target_uvw]

# check number of wires
n_wires = len(source_nodes)
assert (len(target_nodes) == n_wires)
print(f' - Number of wires : {n_wires}')

# instantiate LP problem
prob = pulp.LpProblem("WiresPath", pulp.LpMinimize)

# distance upper bound
max_dist = pulp.LpVariable("MaxDist", lowBound=0.)

# variables
var_dict = {}
cost = {}

# loop over edges
for (i, j) in g.edges:
    # compute edge lengths
    cost[i, j] = np.linalg.norm(g.coords[i] - g.coords[j])
    # binary variable for each edge
    for w in range(n_wires):
        var_dict[i, j, w] = pulp.LpVariable(f'x{w}_{i}_{j}', cat=pulp.LpBinary)

# objective function
prob += max_dist

# constraints
for node in g.nodes:
    in_edges, out_edges = g.get_edges(node)
    for w in range(n_wires):
        in_fluxe = pulp.lpSum([var_dict[i, k, w] for i, k in in_edges])
        out_fluxe = pulp.lpSum([var_dict[k, j, w] for k, j in out_edges])
        if node == source_nodes[w]:
            rhs = -1
        elif node == target_nodes[w]:
            rhs = 1
        else:
            rhs = 0
        prob += in_fluxe - out_fluxe == rhs

    prob += pulp.lpSum([var_dict[i, k, w] for i, k in in_edges for w in range(n_wires)]) <= 1

# prevent crossing edges
for crossing_edges in g.get_all_crossing_edges():
    prob += pulp.lpSum([var_dict[i, j, w] for i, j in crossing_edges for w in range(n_wires)]) <= 1

# distance constraints
d0 = pulp.lpSum([cost[i, j] * var_dict[i, j, 0] for i, j in g.edges])
prob += d0 - max_dist <= 0.
delta_tol = 0.05
for w in range(1, n_wires):
    dw = pulp.lpSum([cost[i, j] * var_dict[i, j, w] for i, j in g.edges])
    prob += dw - max_dist <= 0.
    prob += d0 - dw - delta_tol <= 0.
    prob += d0 - dw + delta_tol >= 0.

# solve
prob.solve()
print(pulp.LpStatus[prob.status])
print(pulp.value(prob.objective))
path = {}
wire_lengths = []
for w in range(n_wires):
    wlen = 0
    path[w] = []
    for (i, j) in g.edges:
        if var_dict[i, j, w].value() == 1.:
            path[w].append((i, j))
            wlen += cost[i, j]
    wire_lengths.append(wlen)
    print(f' - Total lenght for wire {w} = {wlen}')

# plot path
styles = ['b^-', 'go-', 'm*-', 'co-']
fig = plt.figure()
ax = fig.gca(projection='3d')
for w in range(n_wires):
    for (i, j) in path[w]:
        ax.plot([g.coords[i, 0], g.coords[j, 0]],
                [g.coords[i, 1], g.coords[j, 1]],
                [g.coords[i, 2], g.coords[j, 2]],
                styles[w])

    # export to vtk
    export_path_to_vtk(g.coords, path[w], f'wire_{w}.vtp')

plt.show()
