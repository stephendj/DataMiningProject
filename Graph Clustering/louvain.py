import community
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

G = nx.read_edgelist("Data/Dataset 2.csv", delimiter = ";")

partition = community.best_partition(G)

v = defaultdict(list)

for key, value in sorted(partition.items()):
    v[value].append(key)

for key, value in v.items():
	print(key)
	print(value)
	print()

#drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
# for com in set(partition.values()) :
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
#                                 node_color = str(count / size))

# nx.draw_networkx_edges(G,pos, alpha=0.5)
# plt.show()