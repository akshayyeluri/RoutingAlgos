import numpy as np
import cvxpy as cp

def make_prob(adj, R, D):
    n = adj.shape[0]
    mask = (adj == 0)

    phi = cp.Variable((n, n, n), name='phi')
    T = ?
    F = cp.vstack([phi[:, i, :].T @ T[i, :] for i in range(n)])
    obj = cp.Minimize(cp.sum(cp.multiply(D, F)))

    contraints = [0 <= phi, phi <= 1] # Ensure phi values are in 0 to 1 range
    constraints += [phi[:, mask] == 0] # Ensure no nonzero values where no edge
    # Ensure every row of phi[i] is a valid probability distribution,
    # except row i, which must be all zeros
    constraints += [cp.sum(phi, axis=2) == 1 - np.eye(n)]
    # These constraints capture everything but condition 3 on page 75 of 
    # gallagher. This condition is roughly equivalent to mandating certain
    # matrices -- (np.eye(n) - phi[i]) for all i -- be invertible, but idk
    # how to add this constraint in cvxpy (or even if this constraint is convex)

    prob = cp.Problem(obj, constraints)
    return prob




