import numpy as np
import networkx as nx
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
from IPython import embed


class Converter:
    '''
    Given an adjaceny matrix for a network, an instance of this class
    will allow conversion between the 3D array version of phi, and a
    flattened array version of phi representing only the degrees of freedom,
    i.e. the elements that are allowed to be nonzero.

    This is useful for getting the inputs to the objective function
    for scipy to optimize.

    This class also allows converting back, say from the flat array result
    of scipy.optimize to a full 3D array representation of the routing tables
    '''
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
    '''
    This class represents a network, including various state necessary for
    the routing algorithms we are implementing
    '''
    def __init__(self, adj=None, R=np.random.binomial, D_scaling=None, phi=None, seed=7, D_func=None, derivD_func=None, \
                                         n=None, p=0.4):
        self.seed = seed    # Random seed to ensure same graph/results each time

        if adj is not None: # Initialize from adjacency matrix
            self.n = adj.shape[0]
            self.adj = adj
            self.graph = nx.from_numpy_matrix(adj, parallel_edges=False, \
                                                   create_using=nx.DiGraph)
        else: # Initialize from n, p
            self.n = n          # Number of hosts
            inc = 0
            self.graph = nx.gnp_random_graph(n, p, directed=False, seed=seed)
            while not nx.is_connected(self.graph):
                inc += 1 # Update random seed to get a new graph
                self.graph = nx.gnp_random_graph(n, p, directed=False, seed=seed+inc)
            self.graph = self.graph.to_directed()
            # Adjacency matrix of our graph
            self.adj = np.asarray(nx.adjacency_matrix(self.graph).todense())

        if D_scaling is None:
            D_scaling = np.ones((self.n, self.n))
        self.D_scaling = D_scaling # Scaling factors for F_ik's in objective function

        if D_func is None:
            # D_func = lambda F: 1/2 * self.D_scaling * F ** 2
            D_func = lambda F: self.D_scaling * F
        self.D_func = D_func

        if derivD_func is None:
            # derivD_func = lambda F: self.D_scaling * F
            derivD_func = lambda F: self.D_scaling
        self.derivD_func = derivD_func

        # R matrix where R_ij is traffic generated from i going to j
        np.random.seed(seed)
        if callable(R):
            self.R = R(n, p, size=(n,n))
            self.R[np.eye(n).astype(bool)] = 0 # Make sure no traffic from node to itself
        else:
            self.R = R

        # An instance of the converter class for easy use with scipy.optimize
        self.converter = Converter(self.adj)
        self.df = self.converter.degrees_of_freedom

        self.T = None       # T_ij is traffic going through i en route to j
        self.F = None       # F_ik is traffic along edge (i, k)
        # Routing tables, Phi[j, i, k] is fraction of traffic passing through
        # i on the way to j that gets routed to node k next
        self.setPhi(phi)


    def visualize(self, withEdgeTraffic=False, layout='spring', \
            seed=None, label_pos=0.3, ax=None):
        ''' Helper function to visualize networks '''

        if ax is None:
            ax = plt.subplot(111)

        n = self.n
        if layout == 'spring':
            random_seed = self.seed if seed is None else seed
            pos = nx.spring_layout(self.graph, seed=random_seed)
        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)

        nx.draw_networkx(self.graph, pos=pos, ax=ax)
        if withEdgeTraffic and self.checkPhi():
            labels = {(i,j): np.round(self.F[i,j], 3) for i in range(n) for j in range(n) if not np.isclose(self.F[i,j], 0)}
            nx.draw_networkx_edge_labels(self.graph, pos=pos, ax=ax,\
                    edge_labels=labels, label_pos=label_pos)
        return ax



    def setPhi(self, phi):
        ''' This will set the routing tables, and will update the
        F and T matrices using the new routing tables '''
        self.phi = phi
        if self.phi is not None:
            self.T = getTraffic(self.phi, self.R)
            self.F = getF(self.phi, self.T)
            self.derivD = self.derivD_func(self.F)


    def hasPhi(self):
        return (hasattr(self, 'phi') and self.phi is not None)


    def checkPhi(self):
        '''
        This will check if the network has routing tables phi and
        if those tables are valid
        '''
        if self.hasPhi():
            mask = (self.adj == 0)
            cond1 = np.all(self.phi[:, mask] == 0) # Make sure no phi where no edge
            # Make sure valid probability distributions
            cond2 = np.all(np.isclose(np.sum(self.phi, axis=2), \
                                      1 - np.eye(len(self.phi))))
            return (cond1 and cond2)
        return False


    def D_T(self):
        '''
        The expected packet delay of a network, this is the objective function
        we're choosing routing tables to minimize. This is a standalone version
        of the objective, for the one used in optimization routines see
        obj function defined below
        '''
        if not self.hasPhi():
            raise ValueError('No routing tables for network')
        return np.sum(self.D_func(self.F))


    def toPickle(self, fName):
        ''' Dumps the network to a pickle '''
        with open(fName, 'wb') as f:
            pickle.dump((self.adj, self.R, self.D_scaling, self.phi, self.seed), f)


def netFromPickle(fName):
    '''Generate a network from a pickle file'''
    with open(fName, 'rb') as f:
        net = Network(*pickle.load(f))
    return net


def getTraffic(phi, R):
    ''' Get all columns of T by solving n linear systems '''
    n = R.shape[0]
    T = np.empty((n,n))
    for j in range(n):
        A = (np.eye(n) - phi[j, :, :].T) # A_ik = phi_ik(j)
        b = R[:, j]
        if np.linalg.det(A) == 0:
            raise ValueError('LinAlgError caused by invalid phi!')
        T[:, j] = np.linalg.solve(A, b)
    T[np.isclose(T, 0) | (T < 0)] = 0 # Numerical stability
    return T


def getF(phi, T):
    ''' Get the edge traffic amounts from the routing tables and T matrix '''
    n = T.shape[0]
    F = np.empty((n,n))
    for i in range(n):
        F[i, :] = phi[:, i, :].T @ T[i, :]
    F[np.isclose(F, 0) | (F < 0)] = 0 # Numerical stability
    return F
