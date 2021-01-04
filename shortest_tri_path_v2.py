from tri2d import Tri2D, init_square
import pulp
import numpy as np
import matplotlib.pyplot as plt

g = Tri2D(init_square(50, 50))
g.plot()
print(g.tri.simplices)
source_nodes = [0, 20]
target_nodes = [2499, 2449]
obstacle_radius = 0.45
map_center = np.array([0.5, 0.7])

# instantiate
prob = pulp.LpProblem("Shortest_Grid_Path", pulp.LpMinimize)

# binary variable to state a link is chosen or not
var_dict = {}
cost = {}
obstacle = {}
for (i, j) in g.edges:
    x = pulp.LpVariable("x_(%s_%s)" % (i, j), cat=pulp.LpBinary)
    var_dict[i, j] = x
    xy_i = g.coords[i]
    xy_j = g.coords[j]
    cost[i, j] = np.linalg.norm(xy_i - xy_j)
    edge_center = 0.5 * (xy_i + xy_i)
    if np.linalg.norm(edge_center - map_center) < obstacle_radius:
        obstacle[i, j] = 1
    else:
        obstacle[i, j] = 0

obj_dist = pulp.LpVariable("ObjDist")

# objective function
prob += obj_dist

# constraints
for node in g.nodes:
    rhs = 0
    if node in source_nodes:
        rhs = -1
    elif node in target_nodes:
        rhs = 1
    prob += pulp.lpSum([var_dict[i, k] for i, k in g.edges if k == node]) - \
            pulp.lpSum([var_dict[k, j] for k, j in g.edges if k == node]) == rhs

for (i, j) in g.edges:
    prob += var_dict[i, j] * obstacle[i, j] <= 0.5

for node in g.nodes:
    prob += pulp.lpSum([var_dict[i, k] for i, k in g.edges if k == node]) <= 1

# total distance constraint
prob += pulp.lpSum([cost[i, j] * var_dict[i, j] for i, j in g.edges]) - obj_dist <= 0.

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

# plot path
plt.figure()
for (i, j) in path:
    plt.plot([g.coords[i, 0], g.coords[j, 0]],
             [g.coords[i, 1], g.coords[j, 1]], 'k-')
# plot obstacle
theta = np.linspace(-np.pi, np.pi, 100)
x_obstacle = map_center[0] + obstacle_radius * np.cos(theta)
y_obstacle = map_center[1] + obstacle_radius * np.sin(theta)
plt.plot(x_obstacle, y_obstacle)
plt.show()
