from graphs import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



def initializePhi(G):
	phi = np.empty((G.n, G.n, G.n))
	paths = dict(nx.all_pairs_shortest_path(G.graph))
	for j in range(G.n):
		for i in range(G.n):
			if i == j:
				phi[j, i, :] = np.zeros(G.n)
			else:
				k = paths[i][j][1]
				k_vec = np.zeros(G.n)
				k_vec[k] = 1
				phi[j, i, :] = k_vec
	return phi

def phiCheck(phi, G):
    mask = (G.adj == 0)
    cond1 = np.all(phi[:, mask] == 0)
    cond2 = np.all(np.isclose(np.sum(phi, axis=2), 1 - np.eye(len(phi))))
    return (cond1 and cond2)

def calculateMarginals(G, phi, D):
    dR = np.zeros((G.n, G.n))
    for j in range(G.n):
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            for k in range(G.n):
                dR[j, i] += phi[j, i, k] * (D[i, k] + dR[j, k])
    print(dR)

G = Network(6)
phi = initializePhi(G)
D = np.ones((G.n, G.n))
calculateMarginals(G, phi, D)