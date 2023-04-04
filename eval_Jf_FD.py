"""
eval_Jf_FD.py: Use Finite Difference to calculate Jacobian for an arbitrary evalf
"""

import numpy as np
from feval import feval


def eval_Jf_FD(eval_f, x, p, u, variable='x'):
    epsilon = 0.0001
    num_nodes = x.shape[0]

    Jf = np.zeros((num_nodes, num_nodes))
    for k in range(num_nodes):
        e_k = np.zeros(num_nodes)
        e_k[k] = epsilon
        if variable == 'x':
            Jf[:, k] = (feval(eval_f, x + e_k, p, u) - feval(eval_f, x, p, u))/epsilon
        elif variable == 'u':
            e_k = np.column_stack((e_k, np.zeros(num_nodes)))
            Jf[:, k] = (feval(eval_f, x, p, u + e_k) - feval(eval_f, x, p, u))/epsilon
    return Jf
