import numpy as np
import networkx as nx
import scipy.stats as st

class Network:
    def __init__(self, n, p=0.4, R_gen=np.random.binomial):
        self.n = n
        self.p = p

        self.graph = nx.gnp_random_graph(n, p, directed=False)
        while not nx.is_connected(self.graph):
            self.graph = nx.gnp_random_graph(n, p, directed=False)

        self.adj = nx.adjacency_matrix(self.graph).todense()
        self.R = R_gen(n, p, size=(n,n))

        self.T = None
        self.F = None


    def visualize(self, withEdgeTraffic=False, layout='spring', random_seed=7):
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=random_seed)
        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)

        nx.draw_networkx(self.graph, pos=pos)
        if withEdgeTraffic:
            if self.F is None:
                self.updateFT()
            labels = {(i,j): self.F[i,j] for i in range(n) for j in range(n) if self.F[i,j] != 0}
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels)


    def updateF(self):
        self.T = getTraffic(self.Phi, self.R)
        self.F = getF(self.Phi, self.T)


def getTrafficCol(Phi, R, colIdx):
    n = R.shape[0]
    b = R[:, colIdx]
    A = (np.eye(n) - Phi[colIdx, :, :]) # A_ij = phi_ij(colIdx)
    if numpy.linalg.det(A) == 0:
        raise ValueError('LinAlgError caused by invalid Phi!')
    t = np.linalg.solve(A, b)
    return t


def getTraffic(Phi, R):
    n = R.shape[0]
    T = np.empty((n,n))
    for i in range(n):
        T[:, i] = getTrafficCol(Phi, R, i)
    return T


def getF(Phi, T):
    n = T.shape[0]
    F = np.empty((n,n))
    for i in range(n):
        F[i, :] = np.sum(Phi[:, i, :].T @ T[i, :])
    return F
    

    
   
        
        


