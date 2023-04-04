"""
test_jacobian.py: Tests for jacobian.py.
"""

import unittest
import numpy as np
from jacobian import jacobian_fd, jacobian_analytical


class TestJacobian(unittest.TestCase):
    def test_one(self):
        num_nodes = 5

        x = np.ones(num_nodes)
        p = np.ones(num_nodes)
        u = np.ones((num_nodes, 2))

        self.assertTrue(np.allclose(jacobian_fd(x, p, u), jacobian_analytical(x, p, u), 0.0001),
                        'Jacobian functions should match.')

    def test_two(self):
        num_nodes = 6

        x = 2 * np.ones(num_nodes)
        p = 0.5 * np.ones(num_nodes)
        u = 2 * np.ones((num_nodes, 2))

        self.assertTrue(np.allclose(jacobian_fd(x, p, u), jacobian_analytical(x, p, u), 0.0001),
                        'Jacobian functions should match.')

    def test_three(self):
        num_nodes = 7

        x = np.arange(num_nodes)
        p = np.arange(1, num_nodes+1)
        u = 2 * np.ones((num_nodes, 2))

        self.assertTrue(np.allclose(jacobian_fd(x, p, u), jacobian_analytical(x, p, u), 0.0001),
                        'Jacobian functions should match.')
