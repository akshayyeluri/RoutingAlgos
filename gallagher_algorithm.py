from graphs import *
from IPython import embed
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb

def shortestPathsPhi(net):
    '''
    Takes a Network and returns a phi corresponding to routing along the shortest paths (with 
    respect to link length) in the network. Useful for initializing a loop-free phi to start off
    the Gallagher algorithm
    '''
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
    '''
    Takes a Network with n nodes and returns n by n matrix of marginals dR
    (where dR[j,i] is denoted dD_T/dr_i(j) in the Gallagher paper). Utilizes
    the distributed computing method describes in the paper.
    '''
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

def calculateMarginals_v2(net, dR):
    '''
    Takes a Network with n nodes and returns n by n matrix of marginals dR
    (where dR[j,i] is denoted dD_T/dr_i(j) in the Gallagher paper). Utilizes
    linear algebra for speed, is not a distributed algorithm.
    '''
    D = net.D
    n = net.n
    phi = net.phi
    dR = np.zeros((net.n, net.n))
    for j in range(n):
        sub_phi = phi[j]
        b = np.diag(sub_phi @ D.T)
        A = (np.eye(n) - sub_phi)
        dR[j] = np.linalg.solve(A, b)
    return dR

def check_condition(net, j, i, k, dR, eta):
    '''
    Returns True if link (i, k) causes node i to be blocked with respect to j
    '''
    D = net.D
    phi = net.phi
    t = net.T
    is_improper = False
    is_15 = False
    if (phi[j, i, k] > 0) and (dR[j, i] <= dR[j, k]):
        is_improper = True
    if (phi[j, i, k] >= eta * (D[i, k] + dR[j, k] - dR[j, i]) / t[i, j]):
        is_15 = True
    return is_improper and is_15

 
def calculateBlocked(net, dR, eta):
    '''
    Returns matrix tags where tags[j, k] is 1 iff link (i, k) is blocked with respect to j
    for all i.
    '''
    phi = net.phi
    t = net.T
    tags = np.zeros((net.n, net.n))
    for j in range(net.n):
        downstream = {}
        # routing table induced DAG with respect to having destination node j
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            if (i == j):
                downstream[j] = []
                continue
            children = list(destj.successors(i))
            downstream[i] = sum([downstream[c] for c in children], []) + children
            # check if tag has been passed up from children
            if np.any([tags[j, d] for d in downstream[i]]):
                tags[j, i] = True
            else:
                # check condition for immediate downstream children
                for c in children:
                    if check_condition(net, j, i, c, dR, eta):
                        tags[j, i] = True
                    continue
    return tags


def updateRoutingTable(net, dR, tags, eta):
    '''
    Applies Gallagher algorithm for updating phi and returns resultant phi
    '''
    D = net.D
    phi = net.phi
    t = net.T
    for j in range(net.n):
        for i in range(net.n):
            # compute min and min index of D[i, m] + dR[j, m] (over m not blocked)
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
                if (tags[j, k] or net.adj[i, k] == 0):
                    continue
                a = D[i, k] + dR[j, k] - min_D
                delta = min(phi[j, i, k], (eta * a / t[i, j]) if t[i, j] != 0 else np.inf)
                if k != min_ind:
                    # subtract from non optimal links
                    phi[j, i, k] -= delta
                    delta_sum += delta
            # add to optimal link (maintaining probability distribution)
            phi[j, i, min_ind] += delta_sum
    return phi

