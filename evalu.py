"""
evalu.py: Compute u(t) (MIT Tech Shuttle)
"""

import numpy as np


def evalu(t):
    return np.array([[317, 0]] * 3 + [[163, 0], [149, 0]] + [[0, 237]] * 7 + [[225, 0], [175, 0]])
