from graphs import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = Network(6)
#G.visualize()
#plt.show()

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

def phiCheck(phi):
	return np.all(np.isclose(np.sum(phi, axis=2), 1 - np.eye(len(phi))))

G.Phi = initializePhi(G)
G.visualize(withEdgeTraffic = True)
plt.show()