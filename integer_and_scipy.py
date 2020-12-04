import numpy as np
import tqdm
from graphs import *
import itertools
from functools import reduce
from scipy.optimize import minimize
from IPython import embed

def integerRoutingPhi(net):
    '''
    This finds the best integer routing tables for phi by brute force
    searching over all possible integer routing tables
    '''
    n = net.n
    ranges = []
    best_phi = None
    best_score = None

    for dst in range(n):
        for src in range(n):
            inds = np.where(net.adj[src])[0]
            ranges.append(inds)
    n_iter = reduce(lambda x,y: x * y, [len(r) for r in ranges], 1)

    for inds in tqdm.tqdm(itertools.product(*ranges), \
                          desc='Brute Force loop', total=n_iter):
        inds = np.array(inds).reshape(n,n)
        phi = np.zeros((n,n,n))
        for dst, sub_inds in zip(range(n), inds):
            for src, idx in zip(range(n), sub_inds):
                if src == dst:
                    continue
                phi[dst, src, idx] = 1
        try:
            net.setPhi(phi)
            score = net.D_T()
            if best_score is None or score < best_score:
                best_score = score
                best_phi = phi
        except (ValueError, np.linalg.LinAlgError) as e:
            continue

    return best_phi


def obj(phi_flat, network):
    '''
    This calculates D_T, the expected packet delay of a network. This
    version of the objective is used for optimization routines, for a standalone
    D_T version see the D_T method in the Network class
    '''
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


def optim(G):
    '''
    Runs the optimization protocol starting from G.phi on G, returns the best
    results
    '''
    phi0 = G.phi
    x0 = G.converter.fromPhi(phi0)
    #x0 = np.random.rand(G.df)

    bounds = np.zeros((len(x0), 2))
    bounds[:, 1] = 1

    res = minimize(obj, x0=x0, args=(G), bounds=bounds)
    phi_best = G.converter.toPhi(np.round(res.x, 4))
    return res, phi_best



