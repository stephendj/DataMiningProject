library(igraph)
library(MCL)

dat = read.csv("Data/Dataset 2.csv", header = TRUE,sep = ";")
g = graph.data.frame(dat, directed = FALSE)
m = get.adjacency(g)
adjacency = graph.adjacency(m, mode = "undirected")
mcl(adjacency, addLoops = FALSE)
