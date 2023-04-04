"""
trapezoid_dt.py: utilizes trapezoidal method to solve system of ODE with dynamic time-stepping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click

from feval import feval
from eval_Jf_FD import eval_Jf_FD
from visualize import visualize_map


def trapezoid_dt(eval_f, p, u, x_start, t_start, t_stop, timestep, errf=1e-2, errDeltax=1e-2, relDeltax=1e-2, MaxIter=200):
    X = np.array([x_start]).T
    t = [t_start]
    # u = feval(eval_u, t[0])
    dim = 1
    if type(x_start) == np.ndarray:
        dim = x_start.shape[0]

    max_slope = 0
    min_slope = np.inf
    dt = timestep

    for n in range(int(np.ceil((t_stop-t_start)/timestep))):
        # u = feval(eval_u, t[n])
        dt = min(dt, (t_stop - t[n]))
        Xt = np.empty((dim,0))
        k = 0 # Newton iteration index
        Xt = np.column_stack((Xt, X[:,n])) # initial guess is the last point; xt stores intermediate solutions as columns
        gamma = X[:,n] + (dt/2)*feval(eval_f, X[:,n], p, u)
        F = Xt[:,k] - (dt/2)*feval(eval_f, Xt[:,k], p, u) - gamma

        # Set initial errors
        errf_k = np.linalg.norm(F, np.inf)
        dx = 0
        errDeltax_k = 0
        relDeltax_k = 0

        while k < MaxIter and (errf_k > errf or relDeltax_k > relDeltax or errDeltax_k > errDeltax):
            Jf = eval_Jf_FD(eval_f, Xt[:,k], p, u)
            J = np.eye(dim) - (dt/2)*Jf
            dx = np.linalg.lstsq(J, -F)[0]
            Xt = np.column_stack((Xt, Xt[:,k] + dx))
            k += 1
            F = Xt[:,k] - (dt/2)*feval(eval_f, Xt[:,k], p, u) - gamma
            errf_k = np.linalg.norm(F, np.inf)
            errDeltax_k = np.linalg.norm(dx, np.inf)
            relDeltax_k = errDeltax_k/np.linalg.norm(Xt[:,k], np.inf)
        X = np.column_stack((X, Xt[:,k])) # returning only the very last solution
        t.append(t[n] + dt)

        max_slope = max(max_slope, np.linalg.norm(X[:,n+1] - X[:,n], np.inf)/dt)
        min_slope = min(min_slope, np.linalg.norm(X[:,n+1] - X[:,n], np.inf)/dt)
        # print(f"max slope: {max_slope}")
        # print(f"min slope: {min_slope}")

        if np.linalg.norm(X[:,n] - X[:,n-1], np.inf)/dt > max_slope:
            t[n] = max(t[n] - dt, 0)
            n = max(n - 1, 0)
            dt = dt*0.8
            print("reducing dt")
            print(f'n = {n}')
            print(f't[n] = {t[n]}')
            print(f'dt = {dt}')
        elif np.linalg.norm(X[:,n] - X[:,n-1], np.inf)/dt < max_slope:
            t[n] = max(t[n] - dt, 0)
            n = max(n - 1, 0)
            dt = dt*1.2
            print("increasing dt")
            print(f'n = {n}')
            print(f't[n] = {t[n]}')
            print(f'dt = {dt}')
    return t, X

if __name__ == '__main__':
    data = pd.read_csv('line_stop_data.csv')

    # Red line
    r_pm_peak = {}
    num_nodes_rl = 18
    x_rl = np.zeros(num_nodes_rl)
    # p_rl = np.ones(num_nodes_rl) * 2/60  # Made-up transit times
    p_rl = np.array([3, 2, 3, 1, 4, 1, 2, 3, 3, 2, 3, 2, 2, 4, 2, 3, 4, 0]) / 60  # Transit times from MBTA
    u_rl = np.zeros((num_nodes_rl, 2))

    # Orange line
    o_pm_peak = {}
    num_nodes_ol = 20
    x_ol = np.zeros(num_nodes_ol)
    # p_ol = np.ones(num_nodes_ol) * 2/60  # Made-up transit times
    p_ol = np.array([1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 0]) / 60  # Transit times from MBTA
    u_ol = np.zeros((num_nodes_ol, 2))

    for index, row in data.iterrows():
        if row["time_period_name"] == "EVENING" and row["direction_id"] == 1 and row["season"] == "Fall 2019":
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

    times_rl, result_rl = trapezoid_dt('evalf', p_rl, u_rl, x_rl, 0, 2, 0.001, 1e-2, 1e-2, 1e-2, 200)
    np.savetxt('results/TRAP_result_rl.txt', result_rl.T)

    times_ol, result_ol = trapezoid_dt('evalf', p_ol, u_ol, x_ol, 0, 2, 0.001, 1e-2, 1e-2, 1e-2, 200)
    np.savetxt('results/TRAP_result_ol.txt', result_ol.T)

    result_rl = result_rl.T
    result_ol = result_ol.T

    # Uncomment to generate .gif stills
    # num_timesteps = result_rl.shape[0]
    # with click.progressbar(range(num_timesteps)) as bar:
    #     for n in bar:
    #         visualize_map((result_rl[n], p_rl, u_rl), (result_ol[n], p_ol, u_ol), rl_list, ol_list)
    #         plt.title(f'System Outputs (t = {np.round(times_rl[n], 3)}h) (Trapezoidal)')
    #         plt.savefig(f'img/trap_dynamic_vis/vis_{n}.jpg')
    #         plt.close()

    visualize_map((result_rl[-1], p_rl, u_rl), (result_ol[-1], p_ol, u_ol), rl_list, ol_list)
    plt.title('Evening Steady-State (Trapezoidal)')
    plt.show()

    # Uncomment to show dynamics plot
    # for idx, res in enumerate(result_rl):
    #     plt.plot(times_rl, res, label=f'Station {idx + 1}')
    # plt.title('Red Line (Trapezoidal)')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Passengers at Station')
    # plt.legend()
    #
    # plt.figure()
    # for idx, res in enumerate(result_ol):
    #     plt.plot(times_ol, res, label=f'Station {idx + 1}')
    # plt.title('Orange Line (Trapezoidal)')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Passengers at Station')
    # plt.legend()
    # plt.show()
