from collections import defaultdict

import community
import matplotlib.pyplot as plt
import networkx as nx
import time

G = nx.read_edgelist("Dataset2.txt", delimiter = " ")

start_time = time.time()
partition = community.best_partition(G)
end_time = time.time() - start_time

f1 = open('Louvain Result Dataset2.txt', 'w')
f1.write("Time collapsed: " + str(end_time) + "s\n")

v = defaultdict(list)

for key, value in sorted(partition.items()):
    v[value].append(key)

for key, value in v.items():
	f1.write(str(key) + "\n")
	f1.write(str(value) + "\n\n")

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