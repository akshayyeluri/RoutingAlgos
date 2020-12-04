from graphs import *
from IPython import embed
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

eta = 0.01

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

def phiCheck(net):
    phi = net.phi
    mask = (net.adj == 0)
    cond1 = np.all(phi[:, mask] == 0)
    cond2 = np.all(np.isclose(np.sum(phi, axis=2), 1 - np.eye(len(phi))))
    return (cond1 and cond2)

def calculateMarginals(net):
    D = net.D
    phi = net.phi
    dR = np.zeros((net.n, net.n))
    for j in range(net.n):
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            for k in range(net.n):
                dR[j, i] += phi[j, i, k] * (D[i, k] + dR[j, k])
    return dR

# checks if blocking condition is met for link (i, k) with respect to j
def check_condition(net, j, i, k, dR):
    D = net.D
    phi = net.phi
    t = net.T
    is_improper = False
    is_15 = False
    if (phi[j, i, k] > 0) and (dR[j, i] <= dR[j, k]):
        is_improper = True
        # print(is_improper)
    if (phi[j, i, k] >= eta * (D[i, k] + dR[j, k] - dR[j, i]) / t[i, j]):
        is_15 = True
        # print(is_15)
    return is_improper and is_15

# return matrix where tags[j, i] is 1 if i is a blocked node for reaching j
def calculateBlocked(net, dR):
    phi = net.phi
    t = net.T
    tags = np.zeros((net.n, net.n))
    for j in range(net.n):
        downstream = {}
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            if (i == j):
                downstream[j] = []
                continue
            children = list(destj.successors(i))
            downstream[i] = sum([downstream[c] for c in children], []) + children
            if np.any([tags[j, d] for d in downstream[i]]):
                tags[j, i] = True
            else:
                for c in children:
                    if check_condition(net, j, i, c, dR):
                        tags[j, i] = True
                    continue
    return tags


def updateRoutingTable(net, dR, tags):
    D = net.D
    phi = net.phi
    t = net.T
    for j in range(net.n):
        for i in range(net.n):
            min_D = math.inf
            min_ind = -1
            for m in range(net.n):
                if tags[j, m] or (net.adj[i, m] == 0):
                    continue
                if (D[i, m] + dR[j, m] < min_D):
                    min_D = D[i, m] + dR[j, m]
                    min_ind = m
            delta_sum = 0
            for k in range(net.n):
                if (net.adj[i, k] == 0):
                    continue
                a = D[i, k] + dR[j, k] - min_D
                #print(f'a is {a}')
                delta = min(phi[j, i, k], (eta * a / t[i, j]) if t[i, j] != 0 else np.inf)
                #print(f'delta is {delta}')
                #print(f'phi is {phi[j, i, k]}')
                #print('\n')
                if k != m:
                    phi[j, i, k] -= delta
                    delta_sum += delta
            phi[j, i, m] += delta_sum
            #embed()
            print(delta_sum)
    return phi

seed = 6
n = 10
np.random.seed(seed)
D = np.random.randint(1, 3, size=(n,n))
G = Network(n, seed=seed, D=D)
phi = initializePhi(G)
#phi_flat = G.converter.fromPhi(phi)
#phi_flat[phi_flat==0] = 0.2
#phi_flat[phi_flat==1] = 0.8
#G.phi = G.converter.toPhi(phi_flat)
#G.updateFT()
G.phi = phi
score_0 = D_T(G)
for _ in range(7):
    dR = calculateMarginals(G)
    tags = calculateBlocked(G, dR)
    phi = updateRoutingTable(G, dR, tags)
    embed()
    
G.phi = phi
score_1 = D_T(G)










