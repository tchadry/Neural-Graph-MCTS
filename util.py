import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def random_graph(n_nodes = 8):

    # generate complete graph
    graph = nx.complete_graph(n_nodes)
    for node in graph.nodes:
        # add attributes
        graph.nodes[node]['visited'] = False
        graph.nodes[node]['start'] = False
        graph.nodes[node]['current'] = False
        graph.nodes[node]['x'] = random.uniform(0, 1)
        graph.nodes[node]['y'] = random.uniform(0, 1)
    for u, v, data in graph.edges(data=True):
        graph[u][v]['weight'] = np.linalg.norm(
            [graph.nodes[u]['x'] - graph.nodes[v]['x'],
            graph.nodes[u]['y'] - graph.nodes[v]['y']]
        )
    return graph

# print(random_graph())
# nx.draw(random_graph())