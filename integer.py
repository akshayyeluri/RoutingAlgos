import numpy as np
import tqdm
from graphs import *
import itertools
from IPython import embed
from functools import reduce

def integerRouting(net):
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
        net.phi = phi
        try:
            score = D_T(net)
            if best_score is None or score < best_score:
                best_score = score
                best_phi = phi
        except (ValueError, np.linalg.LinAlgError) as e:
            continue

    return best_phi, best_score








