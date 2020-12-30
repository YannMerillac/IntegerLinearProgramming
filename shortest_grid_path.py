from grid2d import Grid2D
import pulp
import numpy as np
import matplotlib.pyplot as plt

def plot_path(g,edges):
    coords = []
    for (i,j) in edges:
        coords.append(g.coords[i,:])
        coords.append(g.coords[j,:])
    coords = np.array(coords)
    plt.figure()
    plt.plot(coords[:,0], coords[:,1])
    plt.show()

g = Grid2D(20,20)
source = 0
target = 54

# instantiate
prob = pulp.LpProblem("Shortest_Grid_Path", pulp.LpMinimize)

# binary variable to state a link is chosen or not
var_dict = {}
cost = {}
for (i, j) in g.edges:
    x = pulp.LpVariable("x_(%s_%s)" % (i,j), cat=pulp.LpBinary)
    var_dict[i, j] = x
    cost[i, j] = np.linalg.norm(g.coords[i] - g.coords[j])

# objective function
prob += pulp.lpSum([cost[i, j] * var_dict[i, j] for i, j in g.edges]), "Total Hop Count"

# constraints
for node in g.nodes:
    rhs = 0
    if node == source:
        rhs = -1
    elif node == target:
        rhs = 1
    prob += pulp.lpSum([var_dict[i, k] for i, k in g.edges if k == node]) - \
            pulp.lpSum([var_dict[k, j] for k, j in g.edges if k == node]) == rhs

# solve
prob.solve()

print(pulp.LpStatus[prob.status])
print(pulp.value(prob.objective))
print("The shortest path is ")
path = []
for link in g.edges:
    if var_dict[link].value() == 1.0:
        print(link, end=" ")
        path.append(link)

plot_path(g,path)