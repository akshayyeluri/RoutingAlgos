from graphs import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def initializePhi(net):
    phi = np.empty((net.n, net.n, net.n))
    paths = dict(nx.all_pairs_shortest_path(net.graph))
    for j in range(net.n):
        for i in range(net.n):
            if i == j:
                phi[j, i, :] = np.zeros(net.n)
            else:
                k = paths[i][j][1]
                k_vec = np.zeros(net.n)
                k_vec[k] = 1
                phi[j, i, :] = k_vec
    return phi

def phiCheck(phi, net):
    mask = (net.adj == 0)
    cond1 = np.all(phi[:, mask] == 0)
    cond2 = np.all(np.isclose(np.sum(phi, axis=2), 1 - np.eye(len(phi))))
    return (cond1 and cond2)

def calculateMarginals(net, phi):
    D = net.D
    dR = np.zeros((net.n, net.n))
    for j in range(net.n):
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            for k in range(net.n):
                dR[j, i] += phi[j, i, k] * (D[i, k] + dR[j, k])
    print(dR)

