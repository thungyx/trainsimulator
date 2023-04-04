"""
visualize.py: Collection of visualization functions for the output of a numerical integrator.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def visualize_graph(line1, line2=None):
    x = line1[0]
    p = line1[1]
    u = line1[2]

    num_nodes = x.shape[0]
    transit_graph = nx.DiGraph()
    if line2 is None:
        transit_graph.add_node('In', pos=((num_nodes-1)/2, 1))
        transit_graph.add_node('Out', pos=((num_nodes - 1) / 2, -1))
    else:
        transit_graph.add_node('In', pos=((num_nodes - 1) / 2 - 1, 2))
        transit_graph.add_node('Out', pos=((num_nodes - 1) / 2 + 1, 2))

    for i in range(1, num_nodes+1):
        transit_graph.add_node(i, pos=(i-1, 0))
        if i != num_nodes:
            transit_graph.add_edge(i, i + 1, w=p[i-1], label=f'{np.round(p[i], 3)} h', color='r')

        u_in = u[i-1, 0]
        u_out = u[i-1, 1]
        if i != num_nodes:
            transit_graph.add_edge('In', i, w=0.002 * u_in, label=f'{np.round(u_in, 3)} pph')
        if i != 1:
            transit_graph.add_edge(i, 'Out', w=0.002 * u_out, label=f'{np.round(u_out, 3)} pph')

    colors = [None, None] + x.tolist()

    if line2 is not None:
        x2 = line2[0]
        p2 = line2[1]
        u2 = line2[2]

        num_nodes2 = x2.shape[0]

        for i in range(1, num_nodes2 + 1):
            new_i = i + num_nodes

            transit_graph.add_node(new_i, pos=(i - 1, 4))
            if i != num_nodes2:
                transit_graph.add_edge(new_i, new_i + 1, w=p2[i-1], label=f'{np.round(p2[i], 3)} h')

            u_in = u2[i - 1, 0]
            u_out = u2[i - 1, 1]
            if i != num_nodes2:
                transit_graph.add_edge('In', new_i, w=0.002 * u_in, label=f'{np.round(u_in, 3)} pph')
            if i != 1:
                transit_graph.add_edge(new_i, 'Out', w=0.002 * u_out, label=f'{np.round(u_out, 3)} pph')

        colors += x2.tolist()

    pos = nx.get_node_attributes(transit_graph, 'pos')
    nodes = nx.draw_networkx_nodes(transit_graph, pos, node_color=colors, cmap=plt.cm.summer_r, node_size=600)
    nx.draw_networkx_labels(transit_graph, pos)

    for edge in transit_graph.edges(data='w'):
        u = edge[0]
        v = edge[1]
        w = edge[2]
        nx.draw_networkx_edges(transit_graph, pos, edgelist=[(u, v)], arrowsize=2 * w, width=0.3 * w)
    nx.draw_networkx_edge_labels(transit_graph, pos, edge_labels=nx.get_edge_attributes(transit_graph, 'label'))
    plt.title('Steady-State Flow')

    color_bar = plt.colorbar(nodes)
    color_bar.set_label('Passengers at Station')


def visualize_map(line1, line2, rl_list, ol_list):
    plt.figure(figsize=(15, 8), dpi=80, tight_layout=True)

    x = line1[0]
    p = line1[1]
    u = line1[2]

    # Draw Red Line
    num_nodes = x.shape[0]
    transit_graph = nx.DiGraph()
    if line2 is None:
        transit_graph.add_node('In', pos=((num_nodes-1)/2, 1))
        transit_graph.add_node('Out', pos=((num_nodes - 1) / 2, -1))
    else:
        transit_graph.add_node('In', pos=((num_nodes - 1) / 2 - 1, 2))
        transit_graph.add_node('Out', pos=((num_nodes - 1) / 2 + 1, 2))

    for i in range(1, num_nodes+1):
        transit_graph.add_node(i, pos=(29.5 - i, 2 * i + 0.5))
        if i != num_nodes:
            transit_graph.add_edge(i, i + 1, w=p[i-1], label=f'{np.round(p[i-1], 3)} h', color='red')

    colors = [None, None] + x.tolist()

    # Draw Orange Line
    x2 = line2[0]
    p2 = line2[1]
    u2 = line2[2]

    num_nodes2 = x2.shape[0]

    for i in range(1, num_nodes2 + 1):
        new_i = i + num_nodes

        transit_graph.add_node(new_i, pos=(2 * i - 1, i + 9))
        if i != num_nodes2:
            transit_graph.add_edge(new_i, new_i + 1, w=p2[i-1], label=f'{np.round(p2[i-1], 3)} h', color='orange')

    colors += x2.tolist()

    pos = nx.get_node_attributes(transit_graph, 'pos')
    nodes = nx.draw_networkx_nodes(transit_graph, pos, node_color=colors, cmap=plt.cm.summer_r, node_size=300)

    labels_rl = {i+1: rl_list[i] for i in range(len(rl_list))}
    labels_ol = {i+1+num_nodes: ol_list[i] for i in range(len(ol_list))}
    labels = {**labels_rl, **labels_ol}

    nx.draw_networkx_labels(transit_graph, pos, labels, font_size=10)

    for edge in transit_graph.edges(data=True):
        u = edge[0]
        v = edge[1]
        w = edge[2]['w']
        if 'color' in edge[2]:
            c = edge[2]['color']
        else:
            c = 'black'
        nx.draw_networkx_edges(transit_graph, pos, edgelist=[(u, v)], edge_color=c, arrowsize=400 * w, width=100 * w)
    # nx.draw_networkx_edge_labels(transit_graph, pos, edge_labels=nx.get_edge_attributes(transit_graph, 'label'))

    color_bar = plt.colorbar(nodes)
    color_bar.set_label('Passengers at Station', size=20)
    color_bar.ax.tick_params(labelsize=20)


def visualize_google_map(rl_data, ol_data, rl_list, ol_list):
    plt.figure(figsize=(12, 8), dpi=80, tight_layout=True)
    img = mpimg.imread('boston_map.png')
    plt.imshow(img, alpha=0.7)

    # Only show labels for terminal stations, Kendall/MIT, and Downtown Crossing
    for i in range(len(rl_list)):
        if (i != 0) and (i != len(rl_list) - 1) and (i != 12):
            rl_list[i] = ''

    for i in range(len(ol_list)):
        if (i != 0) and (i != len(ol_list) - 1) and (i != 10):
            ol_list[i] = ''

    # Set positions of stations on image
    rl_pos = [(502, 679), (494, 612), (497, 563), (453, 495), (447, 489), (396, 359), (388, 332), (388, 290),
              (391, 268), (387, 266), (375, 258), (361, 247), (325, 239), (291, 229), (260, 208), (258, 169),
              (253, 147), (213, 147)]

    ol_pos = [(269, 415), (282, 392), (289, 371), (298, 357), (309, 331), (326, 314), (338, 301), (351, 288),
              (379, 281), (383, 270), (367, 265), (395, 249), (387, 240), (382, 229), (371, 205), (351, 176),
              (350, 160), (349, 136), (352, 60), (359, 27)]

    transit_graph = nx.DiGraph()

    # Draw Red Line
    x_rl = rl_data[0]
    num_nodes_rl = x_rl.shape[0]

    for i in range(1, num_nodes_rl+1):
        transit_graph.add_node(i, pos=rl_pos[i-1])
        if i != num_nodes_rl:
            transit_graph.add_edge(i, i + 1, color='red')

    # Draw Orange Line
    x_ol = ol_data[0]
    num_nodes_ol = x_ol.shape[0]

    for i in range(1, num_nodes_ol + 1):
        new_i = i + num_nodes_rl
        transit_graph.add_node(new_i, pos=ol_pos[i-1])
        if i != num_nodes_ol:
            transit_graph.add_edge(new_i, new_i + 1, color='orange')

    transit_graph.add_node(' ', pos=(-50, 0))
    pos = nx.get_node_attributes(transit_graph, 'pos')
    x_rl[-1] = 0
    x_ol[-1] = 0
    colors = x_rl.tolist() + x_ol.tolist()
    colors.append(1250)
    nodes = nx.draw_networkx_nodes(transit_graph, pos, node_color=colors, cmap=plt.cm.viridis_r, node_size=150)

    labels_rl = {i+1: rl_list[i] for i in range(len(rl_list))}
    labels_ol = {i+1+num_nodes_rl: ol_list[i] for i in range(len(ol_list))}
    labels = {**labels_rl, **labels_ol}
    nx.draw_networkx_labels(transit_graph, pos, labels, font_size=12)

    for edge in transit_graph.edges(data=True):
        u = edge[0]
        v = edge[1]
        c = edge[2]['color']
        nx.draw_networkx_edges(transit_graph, pos, edgelist=[(u, v)], edge_color=c, arrowsize=15, width=4)

    color_bar = plt.colorbar(nodes)
    color_bar.set_label('Passengers at Station', size=20)
    color_bar.ax.tick_params(labelsize=20)
    plt.xlim(0, None)
