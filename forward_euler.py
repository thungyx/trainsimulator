"""
forward_euler.py: Simple forward Euler method ODE solver.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
from evalf import evalf
from visualize import visualize_google_map


def forward_euler(x0, p, u, t_start, t_stop, delta_t):
    x = [x0]
    timestamps = [0]
    t = t_start
    n = 0
    while t < t_stop - delta_t:
        t += delta_t
        n += 1
        x.append(x[n-1] + delta_t * evalf(x[n-1], p, u))
        timestamps.append(timestamps[n-1] + delta_t)

    return timestamps, x


if __name__ == '__main__':
    data = pd.read_csv('line_stop_data.csv')

    # Red line
    r_pm_peak = {}
    num_nodes_rl = 18
    x_rl = np.zeros(num_nodes_rl) + 500
    # p_rl = np.ones(num_nodes_rl) * 2/60  # Made-up transit times
    p_rl = np.array([3, 2, 3, 1, 4, 1, 2, 3, 3, 2, 3, 2, 2, 4, 2, 3, 4, 0]) / 60  # Transit times from MBTA
    u_rl = np.zeros((num_nodes_rl, 2))

    # Orange line
    o_pm_peak = {}
    num_nodes_ol = 20
    x_ol = np.zeros(num_nodes_ol) + 500
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

    times_rl, result_rl = forward_euler(x_rl, p_rl, u_rl, 0, 2, 5e-4)
    result_rl = np.row_stack(result_rl)
    np.savetxt('results/FE_result_rl.txt', result_rl)

    times_ol, result_ol = forward_euler(x_ol, p_ol, u_ol, 0, 2, 5e-4)
    result_ol = np.row_stack(result_ol)
    np.savetxt('results/FE_result_ol.txt', result_ol)

    # Uncomment to generate .gif stills
    # num_timesteps = result_rl.shape[0]
    # dec = 50
    # with click.progressbar(range(num_timesteps)) as bar:
    #     for n in bar:
    #         if n % dec != 0:
    #             continue
    #         visualize_google_map((result_rl[n], p_rl, u_rl), (result_ol[n], p_ol, u_ol), rl_list, ol_list)
    #         plt.title(f'Evening - System Outputs (t = {np.round(times_rl[n], 1)}h) (Forward Euler)', fontsize=20)
    #         plt.savefig(f'img/fe_dynamic_vis_2/vis_{n//dec:03d}.jpg')
    #         plt.close()

    visualize_google_map((result_rl[-1], p_rl, u_rl), (result_ol[-1], p_ol, u_ol), rl_list, ol_list)
    plt.title(r'Evening Steady-State (Forward Euler, $\Delta t = 5 \cdot 10^{-4} h$)', fontsize=30)
    plt.show()

    # Uncomment to show dynamic plots
    # plt.figure(figsize=(5, 3), tight_layout=True)
    # for n in range(result_rl.shape[1] - 1):
    #     plt.plot(times_rl, result_rl[:, n], label=f'Station {n+1}')
    #
    # plt.title(r'Evening - Red Line (FE, $\Delta t = 5 \cdot 10^{-4}$ h)')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Passengers at Station')
    # plt.xlim(0, 2)
    # plt.ylim(0, 1800)
    #
    # plt.figure(figsize=(5, 3), tight_layout=True)
    # for n in range(result_ol.shape[1] - 1):
    #     plt.plot(times_ol, result_ol[:, n], label=f'Station {n+1}')
    #
    # plt.title(r'Evening - Orange Line (FE, $\Delta t = 5 \cdot 10^{-4}$ h)')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Passengers at Station')
    # plt.xlim(0, 2)
    # plt.ylim(0, 1800)
    # plt.show()
