import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

color_map = {
    'B': 'tab:olive',
    'M': 'tab:olive',
    'H': 'tab:green',
    'Q': 'tab:red',
    'Y': 'tab:blue',
    'X': 'tab:blue',
    'P': 'tab:orange',
    'A': 'tab:purple',
    'D': 'tab:cyan',
    'S': 'tab:brown',
    'R': 'tab:pink',
    'T': 'tab:gray',
}


def names_remapping(uniques):
    mapping = {}
    for f in uniques:
        if 'B' in f:
            mapping[f] = 'B'
        if 'H' in f:
            mapping[f] = 'H'
        if 'Q' in f:
            mapping[f] = 'Q'
        if 'Y' in f:
            mapping[f] = 'Y'
        if 'X' in f:
            mapping[f] = 'X'
        if 'P' in f:
            mapping[f] = 'P'
        if 'A' in f:
            mapping[f] = 'A'
        if 'D' in f:
            mapping[f] = 'D'
        if 'S' in f:
            mapping[f] = 'S'
        if 'M' in f:
            mapping[f] = 'M'
        if 'R' in f:
            mapping[f] = 'R'
        if 'T' in f:
            mapping[f] = 'T'

    return mapping


def plotter_netx(edges):
    uniques = np.unique(edges)
    mapping = names_remapping(uniques)

    G = nx.Graph()
    G.add_nodes_from([(node, {'name': attr}) for (node, attr) in mapping.items()])
    G.add_edges_from(edges)

    c_colors = []
    for n in G:
        if (G.nodes[n]['name'] in color_map):
            c_colors.append(f'{color_map[G.nodes[n]["name"]]}')
        else:
            c_colors.append('gray')

    labels = nx.get_node_attributes(G, 'name')
    nx.draw(G, labels=labels, with_labels=True, font_size=22, node_size=800, node_color=c_colors,
            font_color="whitesmoke")  # , node_size=1000 , font_weight='bold'


    return 0

