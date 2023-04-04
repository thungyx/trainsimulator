import numpy as np

from evalf import evalf
from jacobian import jacobian_fd


def linearize(x0, p, u0):
    J0_f = jacobian_fd(x0, p, u0)
    A = J0_f
    J0_u = jacobian_fd(x0, p, u0, variable='u')
    K0 = evalf(x0, p, u0) - J0_f @ x0 - J0_u @ (u0 @ [1, -1])
    B = np.column_stack((K0, J0_u))

    return A, B
