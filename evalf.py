"""
evalf.py: Compute f(x, p, u) in dx/dt = f(x, p, u).
"""

import numpy as np

# P_max = 1200 * 15  # Max. 1200 passengers per trip, 10 trips per hour
P_max = 1200 * 20  # Max. 1200 passengers per trip, 20 trips per hour


def evalf(x, p, u):
    num_nodes = x.shape[0]

    f = np.zeros(num_nodes)
    for node in range(num_nodes):
        f[node] = u[node] @ [1, -1]

        if node != num_nodes - 1:
            f[node] -= P_max * np.tanh(x[node]/(P_max * p[node]))

        if node != 0:
            f[node] += P_max * np.tanh(x[node-1]/(P_max * p[node-1]))

    return f
