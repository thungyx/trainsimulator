"""
jacobian.py: Use Finite Difference and analytical methods to calculate Jacobian for evalf.py.
"""

import numpy as np
from evalf import evalf, P_max


def jacobian_fd(x, p, u, variable='x'):
    epsilon = 0.0001
    num_nodes = x.shape[0]

    Jf = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        e_k = np.zeros(num_nodes)
        e_k[k] = epsilon
        if variable == 'x':
            Jf[:, k] = (evalf(x + e_k, p, u) - evalf(x, p, u))/epsilon
        elif variable == 'u':
            e_k = np.column_stack((e_k, np.zeros(num_nodes)))
            Jf[:, k] = (evalf(x, p, u + e_k) - evalf(x, p, u)) / epsilon
    return Jf


def jacobian_analytical(x, p, u):
    num_nodes = x.shape[0]

    J_f = np.zeros((num_nodes, num_nodes))

    for k in range(num_nodes):
        J_f[k, k] = -1/(p[k] * np.cosh(2*x[k]/(P_max * p[k])))

        if k != 0:
            J_f[k, k-1] = 1/(p[k-1] * np.cosh(2*x[k-1]/(P_max * p[k-1])))
    return J_f
