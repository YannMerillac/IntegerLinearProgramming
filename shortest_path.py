import networkx as nx
import pulp
import random
import matplotlib.pyplot as plt

g = nx.to_directed(nx.barabasi_albert_graph(20, 2))
nx.draw(g, with_labels=True)
plt.show()
source = 0
target = 13

dict_d = {}
for i, j in g.edges:
    dict_d[i, j] = dict_d[j, i] = round(random.uniform(1.0, 20.0), 2)

nx.set_edge_attributes(g, dict_d, 'delay')

# instantiate
prob = pulp.LpProblem("Shortest Path Problem", pulp.LpMinimize)
cost = nx.get_edge_attributes(g, 'delay')

# binary variable to state a link is chosen or not
var_dict = {}
for (i, j) in g.edges:
    x = pulp.LpVariable("x_(%s_%s)" % (i,j), cat=pulp.LpBinary)
    var_dict[i, j] = x

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
for link in g.edges:
    if var_dict[link].value() == 1.0:
        print(link, end=" ")