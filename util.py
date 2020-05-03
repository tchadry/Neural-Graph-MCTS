import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def random_graph(n_nodes=8):

    # generate complete graph
    graph = nx.complete_graph(n_nodes)
    for node in graph.nodes:
        # add attributes
        graph.nodes[node]['visited'] = False
        graph.nodes[node]['start'] = False
        graph.nodes[node]['current'] = False
        graph.nodes[node]['x_pos'] = random.uniform(0, 1)
        graph.nodes[node]['y_pos'] = random.uniform(0, 1)
    for u, v, data in graph.edges(data=True):
        graph[u][v]['weight'] = np.linalg.norm(
            [graph.nodes[u]['x_pos'] - graph.nodes[v]['x_pos'],
            graph.nodes[u]['y_pos'] - graph.nodes[v]['y_pos']]
        )
    return graph

# print(random_graph())
# nx.draw(random_graph())

# first we define the average meter, which is class used in alphazero


class AverageMeter(object):
    """
     Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
  
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count


class dotdict(dict):

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(dotdict, self).__setitem__(key, value)
        self.__dict__.update({key: value})
