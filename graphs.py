import numpy as np
import networkx as nx
import scipy.stats as st
from IPython import embed
import matplotlib.pyplot as plt

# TODO: 
# initialize graph with adjacency matrix
# fix visualization (directed graph)
# 

class Converter:
    def __init__(self, adj):
        self.n = adj.shape[0]
        self.mask = self.genMask(adj)
        self.degrees_of_freedom = np.sum(self.mask)

    def genMask(self, adj):
        mask0 = adj.copy()
        mask = np.tile(mask0, (self.n, 1, 1))
        for i in range(self.n):
            mask[i, i, :] = False
        return mask.astype(bool)

    def toPhi(self, flat_arr):
        phi = np.zeros((self.n, self.n, self.n))
        phi[self.mask] = flat_arr
        tots = np.sum(phi, axis=2)
        tots[np.eye(self.n).astype(bool)] = 1
        phi = phi / tots[:, :, None]
        return phi

    def fromPhi(self, phi):
        return phi[self.mask]


class Network:
    def __init__(self, n, p=0.4, R_gen=np.random.binomial, seed=7, D=None):
        self.n = n
        self.p = p
        self.seed = seed

        if D is None:
            D = np.ones((n,n))
        self.D = D

        inc = 0
        self.graph = nx.gnp_random_graph(n, p, directed=False, seed=seed)
        while not nx.is_connected(self.graph):
            inc += 1
            self.graph = nx.gnp_random_graph(n, p, directed=False, seed=seed+inc)

        self.adj = np.asarray(nx.adjacency_matrix(self.graph).todense())
        np.random.seed(seed)
        self.R = R_gen(n, p, size=(n,n))
        self.R[np.eye(n).astype(bool)] = 0 # Make sure no traffic from node to itself
        self.converter = Converter(self.adj)
        self.df = self.converter.degrees_of_freedom

        self.T = None
        self.F = None
        self.phi = None


    def visualize(self, withEdgeTraffic=False, layout='spring', \
            seed=None, label_pos=0.3, ax=None):

        if ax is None:
            ax = plt.subplot(111)

        n = self.n
        if layout == 'spring':
            random_seed = self.seed if seed is None else seed
            pos = nx.spring_layout(self.graph, seed=random_seed)
        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)

        nx.draw_networkx(self.graph, pos=pos, ax=ax)
        if withEdgeTraffic:
            self.updateFT()
            labels = {(i,j): self.F[i,j] for i in range(n) for j in range(n) if self.F[i,j] != 0}
            nx.draw_networkx_edge_labels(self.graph, pos=pos, ax=ax,\
                    edge_labels=labels, label_pos=label_pos)
        return ax


    def updateFT(self):
        self.T = getTraffic(self.phi, self.R)
        self.F = getF(self.phi, self.T)


def getTrafficCol(phi, R, colIdx):
    n = R.shape[0]
    b = R[:, colIdx]
    A = (np.eye(n) - phi[colIdx, :, :]) # A_ij = phi_ij(colIdx)
    if np.linalg.det(A) == 0:
        raise ValueError('LinAlgError caused by invalid phi!')
    t = np.linalg.solve(A, b)
    return t


def getTraffic(phi, R):
    n = R.shape[0]
    T = np.empty((n,n))
    for i in range(n):
        T[:, i] = getTrafficCol(phi, R, i)
    return T


def getF(phi, T):
    n = T.shape[0]
    F = np.empty((n,n))
    for i in range(n):
        F[i, :] = phi[:, i, :].T @ T[i, :]
    return F


def D_T(network):
    network.updateFT()
    return np.sum(network.D * network.F)


def obj(phi_flat, network):
    n = network.n
    R = network.R
    D = network.D
    conv = network.converter

    phi = conv.toPhi(phi_flat)
    try:
        T = getTraffic(phi, R)
    except ValueError as e:
        return np.inf
    F = getF(phi, T)
    return np.sum(D * F)

   

############################################################
# Deprecated
############################################################

#class Converter:
#    def __init__(self, adj):
#        self.n = adj.shape[0]
#        mask, first = self.genMaskAndFirst(adj)
#        self.mask = mask
#        self.first = first
#        self.degrees_of_freedom = np.sum(self.mask)
#
#
#    def genMaskAndFirst(self, adj):
#        first_inds = (np.arange(self.n), np.array([np.where(row)[-1][0] for row in adj]))
#        mask0 = adj.copy()
#        mask0[first_inds] = False
#        mask = np.tile(mask0, (self.n, 1, 1))
#        for i in range(self.n):
#            mask[i, i, :] = False
#        return mask.astype(bool), first_inds
#
#
#    def toPhi(self, flat_arr):
#        phi = np.zeros((self.n, self.n, self.n))
#        phi[self.mask] = flat_arr
#        residual = (1 - np.eye(self.n)) - np.sum(phi, axis=2)
#        for j in range(self.n):
#            sub_arr = phi[j]
#            sub_arr[self.first] = residual[j, :]
#        return phi
#
#
#    def fromPhi(self, phi):
#        return phi[self.mask]
        


