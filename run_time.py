"""
run_time.py script to check time it takes
"""


import time
import pandas as pd
import numpy as np

from statistics import mean
import matplotlib.pyplot as plt
from evalf import evalf
from forward_euler import forward_euler
from trapezoid import trapezoid
from trapezoid_dt import trapezoid_dt

# Obtain inputs
data = pd.read_csv('line_stop_data.csv')

# Red line
r_pm_peak = {}
num_nodes_rl = 18
x_rl = np.zeros(num_nodes_rl)
p_rl = np.ones(num_nodes_rl) * 2/60
u_rl = np.zeros((num_nodes_rl, 2))

# Orange line
o_pm_peak = {}
num_nodes_ol = 20
x_ol = np.zeros(num_nodes_ol)
p_ol = np.ones(num_nodes_ol) * 2/60
u_ol = np.zeros((num_nodes_ol, 2))

for index, row in data.iterrows():
    if row["time_period_name"] == "AM_PEAK" and row["direction_id"] == 1 and row["season"] == "Fall 2019":
        if row["route_id"] == "Red":
            r_pm_peak[row["stop_name"]] = [row["average_ons"], row["average_offs"]]
        elif row["route_id"] == "Orange":
            o_pm_peak[row["stop_name"]] = [row["average_ons"], row["average_offs"]]

rl_list = ['Braintree', 'Quincy Adams', 'Quincy Center', 'Wollaston', 'North Quincy', 'JFK/Umass', 'Andrew',
           'Broadway', 'South Station', 'Downtown Crossing', 'Park Street', 'Charles/MGH', 'Kendall/MIT', 'Central',
           'Harvard', 'Porter', 'Davis', 'Alewife']
for idx, name in enumerate(rl_list):
    u_rl[idx] = r_pm_peak[name]

ol_list = ['Forest Hills', 'Green Street', 'Stony Brook', 'Jackson Square', 'Roxbury Crossing', 'Ruggles',
           'Massachusetts Ave.', 'Back Bay', 'Tufts Medical Center', 'Chinatown', 'Downtown Crossing',
           'State Street', 'Haymarket', 'North Station', 'Community College', 'Sullivan Square', 'Assembly',
           'Wellington', 'Malden Center', 'Oak Grove']
for idx, name in enumerate(ol_list):
    u_ol[idx] = o_pm_peak[name]

# Hyperparameters
timestep = 1e-1
ode = "forward_euler"
abbr = "FE"
full = "Forward Euler"
# ode = "trapezoid"
# abbr = "TRAP"
# full = "Trapezoidal Method"
ode = "trapezoid_dt"
abbr = "TRAP_dt"
full = "Trapezoidal Method (Dynamic)"


t = []
if ode == "forward_euler":
    for j in range(1):
        start = time.time()
        times_rl, result_rl = forward_euler(x_rl, p_rl, u_rl, 0, 2, timestep)
        times_ol, result_ol = forward_euler(x_ol, p_ol, u_ol, 0, 2, timestep)
        elapsed = time.time()-start
        t.append(elapsed)
    print(f'Average run-time {ode} for {timestep} is {mean(sorted(t)[:5])}')
    result_rl = np.row_stack(result_rl)
    np.savetxt(f'results/{abbr}_result_rl_{timestep}.txt', result_rl)
    result_ol = np.row_stack(result_ol)
    np.savetxt(f'results/{abbr}_result_ol_{timestep}.txt', result_ol)

    plt.figure()
    for n in range(result_rl.shape[1]):
        plt.plot(times_rl, result_rl[:, n], label=f'Station {n + 1}')

    plt.title(f'Red Line ({full})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Passengers at Station')
    plt.legend()

    plt.figure()
    for n in range(result_ol.shape[1]):
        plt.plot(times_ol, result_ol[:, n], label=f'Station {n + 1}')

    plt.title(f'Orange Line ({full})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Passengers at Station')
    plt.legend()

    plt.show()

if ode == "trapezoid" or ode == "trapezoid_dt":
    if ode == "trapezoid":
        for j in range(20):
            start = time.time()
            times_rl, result_rl = trapezoid('evalf', p_rl, u_rl, x_rl, 0, 2, timestep, 1e-2, 1e-2, 1e-2, 20)
            times_ol, result_ol = trapezoid('evalf', p_ol, u_ol, x_ol, 0, 2, timestep, 1e-2, 1e-2, 1e-2, 20)
            elapsed = time.time()-start
            t.append(elapsed)
        print(f'Average run-time {ode} for {timestep} is {mean(sorted(t)[:5])}')
    if ode == "trapezoid_dt":
        for j in range(1):
            start = time.time()
            times_rl, result_rl = trapezoid_dt('evalf', p_rl, u_rl, x_rl, 0, 2, timestep, 1e-2, 1e-2, 1e-2, 200)
            times_ol, result_ol = trapezoid_dt('evalf', p_ol, u_ol, x_ol, 0, 2, timestep, 1e-2, 1e-2, 1e-2, 200)
            elapsed = time.time()-start
            t.append(elapsed)
        print(f'Average run-time {ode} for {timestep} is {mean(sorted(t)[:5])}')
        np.savetxt(f'results/{abbr}_times_rl_{timestep}.txt', times_rl)
        np.savetxt(f'results/{abbr}_times_ol_{timestep}.txt', times_ol)
    np.savetxt(f'results/{abbr}_result_rl_{timestep}.txt', result_rl.T)
    np.savetxt(f'results/{abbr}_result_ol_{timestep}.txt', result_ol.T)

    for idx, res in enumerate(result_rl):
        plt.plot(times_rl, res, label=f'Station {idx + 1}')
    plt.title(f'Red Line ({full})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Passengers at Station')
    plt.legend()

    plt.figure()
    for idx, res in enumerate(result_ol):
        plt.plot(times_ol, res, label=f'Station {idx + 1}')
    plt.title(f'Orange Line ({full})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Passengers at Station')
    plt.legend()
    plt.show()
