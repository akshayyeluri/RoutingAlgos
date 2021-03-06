from graphs import *
from IPython import embed
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb

def shortestPathsPhi(net, useCurrentF=False):
    '''
    Takes a Network and returns a phi corresponding to routing along the shortest paths (with
    respect to link length) in the network. Useful for initializing a loop-free phi to start off
    the Gallagher algorithm
    '''
    if not useCurrentF:
        paths = dict(nx.all_pairs_dijkstra_path(net.graph))
    else:
        w = {(i, j): net.F[i, j] for i in range(net.n) for j in range(net.n)}
        g = nx.from_numpy_matrix(net.F, create_using=nx.DiGraph)
        nx.set_edge_attributes(g, w, 'weight')
        paths = dict(nx.all_pairs_dijkstra_path(g, weight='weight'))

    phi = np.empty((net.n, net.n, net.n))
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
    D = net.derivD
    phi = net.phi
    dR = np.zeros((net.n, net.n))
    for j in range(net.n):
        destj = nx.from_numpy_matrix(phi[j, :, :], parallel_edges=False, create_using=nx.DiGraph)
        top_sorted = list(reversed(list(nx.topological_sort(destj))))
        for i in top_sorted:
            for k in range(net.n):
                dR[j, i] += phi[j, i, k] * (D[i, k] + dR[j, k])
    return dR

def calculateMarginals_v2(net):
    '''
    Takes a Network with n nodes and returns n by n matrix of marginals dR
    (where dR[j,i] is denoted dD_T/dr_i(j) in the Gallagher paper). Utilizes
    linear algebra for speed, is not a distributed algorithm.
    '''
    D = net.derivD
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
    D = net.derivD
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
    D = net.derivD
    phi = net.phi
    t = net.T
    for j in range(net.n):
        for i in range(net.n):

            # Vectorized version
            #mask = (tags[j,:] == 0) & (net.adj[i,:] > 0)
            #ms = np.where(mask)[0]
            #min_idx = ms[np.argmin(D[i, mask] + dR[j, mask])]
            #min_D = D[i, min_idx] + dR[j, min_idx]
            #a = D[i, mask] + dR[j, mask] - min_D
            #delt = np.zeros(net.n)
            #delt[mask] = np.minimum(phi[j, i, mask], (eta * a / t[i, j]) if t[i, j] != 0 else np.inf)
            #idx = np.ones(net.n, bool); idx[min_idx] = False
            #phi[j, i, min_idx] += np.sum(delt[idx])
            #phi[j, i, idx] -= delt[idx]
            #phi[j, i, ~mask] = 0

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



def convergenceConditions(net):
    ''' Checks condition 8 in Gallagher Paper to see if we have valid convergence '''
    dR = calculateMarginals(net)
    phi = net.phi
    D = net.derivD
    n = net.n

    return all([(D[i, k] + dR[j, k] - dR[j, i] >= 0) \
                 for i,j,k, in zip(range(n), range(n), range(n))])


def iterAlgo(net, updateFunc, nTrials=None, converge_perc=0.1, retPhi=False, **kwargs):
    '''
    Iteratively runs a routing algorithm given by updateFunc with kwargs arguments on
    net, returning all the scores for each iteration.

    Converges after nTrials (if nTrials is not None),
    or after the last converge_perc fraction of the scores are close
    '''
    assert(net.hasPhi())
    scores = [net.D_T()]

    if retPhi:
        phis = [net.phi.copy()]

    while True:
        phi = updateFunc(net, **kwargs)
        net.setPhi(phi)
        scores.append(net.D_T())

        if retPhi:
            phis.append(phi.copy())

        # Convergence if nTrials reached
        if nTrials is not None:
            if len(scores) == nTrials + 1:
                break

        # Convergence if last 10% of scores all close
        else:
            n_scores = len(scores)
            n_thresh = int(n_scores * converge_perc)
            last_scores = np.array(scores[n_scores - n_thresh:])
            if (n_scores > converge_perc * 200) and \
                    np.all(np.isclose(last_scores, scores[-1])):
                break

    if retPhi:
        return scores, phis
    return scores



def iterGallagher(net, eta=0.1, nTrials=None, converge_perc=0.1, retPhi=False):
    ''' Iteratively run the Gallagher algorithm '''

    def updateFunc(net, eta):
        dR = calculateMarginals(net)
        tags = calculateBlocked(net, dR, eta)
        return updateRoutingTable(net, dR, tags, eta)

    return iterAlgo(net, updateFunc=updateFunc, nTrials=nTrials, converge_perc=converge_perc,
                    retPhi=retPhi, eta=eta)
