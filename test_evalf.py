"""
test_evalf.py: Tests for evalf.py.
"""

import unittest
import numpy as np
from evalf import evalf


class TestEvalf(unittest.TestCase):
    def test_zero_flow(self):
        num_nodes = 5

        x = np.zeros(num_nodes)
        p = np.ones(num_nodes)
        u = np.zeros((num_nodes, 2))

        self.assertTrue((evalf(x, p, u) == np.zeros(num_nodes)).all(),
                        'No passengers should be at station in zero case.')

    def test_one(self):
        num_nodes = 2

        x = np.zeros(num_nodes)
        p = np.ones(num_nodes)
        u = np.eye(num_nodes)

        self.assertTrue((evalf(x, p, u) == np.array([1, -1])).all())

    def test_two(self):
        num_nodes = 2

        x = np.ones(num_nodes)
        x[-1] = 0
        p = np.ones(num_nodes)
        u = np.eye(num_nodes)

        self.assertTrue(np.allclose(evalf(x, p, u), [1.33e-6, -1.33e-6]))

    def test_three(self):
        num_nodes = 3

        x = np.ones(num_nodes)
        p = 2*np.ones(num_nodes)
        u = np.array([[4, 0], [3, 1], [0, 6]])

        self.assertTrue(np.allclose(evalf(x, p, u), [3.5, 2, -6]))
