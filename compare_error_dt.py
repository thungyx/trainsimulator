"""
compare_error_dt.py: Find maximum difference between Forward Euler and Trapezoidal (Dynamic dt) outputs.
"""

import numpy as np
import matplotlib.pyplot as plt

timestep = 0.001

FE_result_rl = np.loadtxt('results/FE_result_rl_1e-06.txt')
TRAP_result_rl = np.loadtxt(f'results/TRAP_dt_result_rl_{timestep}.txt')
times_rl = np.loadtxt(f'results/TRAP_dt_times_rl_{timestep}.txt')

err_rl = FE_result_rl[(1e6 * times_rl).astype(int), :] - TRAP_result_rl
# max_err_rl = np.linalg.norm(err_rl, ord=np.inf)
max_err_rl = np.max(abs(err_rl))
print(f'Red Line: Max. Error across Stations/Timesteps: {np.round(max_err_rl, 5)}')

FE_result_ol = np.loadtxt('results/FE_result_ol_1e-06.txt')
TRAP_result_ol = np.loadtxt(f'results/TRAP_dt_result_ol_{timestep}.txt')
times_ol = np.loadtxt(f'results/TRAP_dt_times_ol_{timestep}.txt')

err_ol = FE_result_ol[(1e6 * times_ol).astype(int), :] - TRAP_result_ol
# max_err_ol = np.linalg.norm(err_ol, ord=np.inf)
max_err_ol = np.max(abs(err_ol))
print(f'Orange Line: Max. Error across Stations/Timesteps: {np.round(max_err_ol, 5)}')

# Plot graph of errors
for idx, res in enumerate(err_rl.T):
    plt.plot(times_rl, res, label=f'Station {idx + 1}')
plt.title('Red Line')
plt.xlabel('Time (hours)')
plt.ylabel('Error (Passengers at Station)')
plt.legend()

plt.figure()
for idx, res in enumerate(err_ol.T):
    plt.plot(times_ol, res, label=f'Station {idx + 1}')
plt.title('Orange Line')
plt.xlabel('Time (hours)')
plt.ylabel('Error (Passengers at Station)')
plt.legend()
plt.show()
