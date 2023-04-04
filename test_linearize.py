import numpy as np
import matplotlib.pyplot as plt

from evalf import evalf
from linearize import linearize


num_nodes = 1
p = np.ones(num_nodes)
u = np.array([[2, 1]])
x_in = np.arange(0, 2000)
f_out = [evalf(x.reshape(1, 1), p, u) for x in x_in]

x0 = np.array([1000])
a, b = linearize(x0, p, u)
u_net = u[0, 0] - u[0, 1]
f_lin_out = [a[0] * x + b[0] @ [1, u_net] for x in x_in]

plt.plot(x_in, f_out, label='Actual f(x, p, u)')
plt.plot(x_in, f_lin_out, label='Linearized f(x, p, u)')
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.legend()
plt.show()
