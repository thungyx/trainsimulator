"""
compare_error.py: Find maximum difference between Forward Euler and Trapezoidal outputs.
"""

import numpy as np
import matplotlib.pyplot as plt

timestep = 0.001

FE_result_rl = np.loadtxt('results/FE_result_rl_1e-06.txt')
TRAP_result_rl = np.loadtxt('results/TRAP_dt_result_rl_0.001.txt')

err_rl = FE_result_rl[[int(timestep / 1e-6) * i for i in range(TRAP_result_rl.shape[0])], :] - TRAP_result_rl
# max_err_rl = np.linalg.norm(err_rl, ord=np.inf)
max_err_rl = np.max(abs(err_rl))
print(f'Red Line: Max. Error across Stations/Timesteps: {np.round(max_err_rl, 5)}')

FE_result_ol = np.loadtxt('results/FE_result_ol_1e-06.txt')
TRAP_result_ol = np.loadtxt('results/TRAP_dt_result_ol_0.001.txt')

err_ol = FE_result_ol[[int(timestep / 1e-6) * i for i in range(TRAP_result_ol.shape[0])], :] - TRAP_result_ol
# max_err_ol = np.linalg.norm(err_ol, ord=np.inf)
max_err_ol = np.max(abs(err_ol))
print(f'Orange Line: Max. Error across Stations/Timesteps: {np.round(max_err_ol, 5)}')

# Plot graph of errors
times = np.linspace(0, 2, int(2 / timestep) + 1)
for idx, res in enumerate(err_rl.T):
    plt.plot(times, res, label=f'Station {idx + 1}')
plt.title('Red Line')
plt.xlabel('Time (hours)')
plt.ylabel('Error (Passengers at Station)')
plt.legend()

plt.figure()
for idx, res in enumerate(err_ol.T):
    plt.plot(times, res, label=f'Station {idx + 1}')
plt.title('Orange Line')
plt.xlabel('Time (hours)')
plt.ylabel('Error (Passengers at Station)')
plt.legend()
plt.show()
