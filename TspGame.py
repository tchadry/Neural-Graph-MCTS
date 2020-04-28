from torch_geometric.utils.convert import from_networkx
from itertools import permutations
import util
import numpy as np
import torch

class TspGame():

    def __init__(self, n):
        self.n = n
        self.graph = util.random_graph(n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """

        return self.n

    def getNextState(self, path, action):
        """
        Input:
            current path as list of index
            action to take as node index

        Returns:
            path and pay of move (negative of distance between 2 nodes)
        """

        pay = -(self.graph[path[-1]][action]['weight'])
        if len(path) == self.n - 1:
            pay -= self.graph[path[0]][action]['weight']
        return (path + [action], pay)

    def getGameEnded(self, path):
        """
        Returns:
            bool if path is the full cycle
               
        """
        return len(path) == self.n

    def construct_graph(self, path):
        '''
        Input: list of nodes as a path

        Returns: torch geometric graph input Data
        '''

        # CHECK THIS PART AGAIN!!!!
        current_graph = self.graph.copy()
        current_graph.nodes[0]['start'] = True
        current_graph.nodes[path[-1]]['current'] = False
        for node in path:
            current_graph.nodes[node]['visited'] = True

        G = from_networkx(current_graph)
        G.x = torch.stack([G.visited.float(),
                        G.first.float(),
                        G.current.float(),
                        G.x.float(),
                        G.y.float()]).T

        G.current = None
        G.first = None
        G.visited = None
        G.x = None
        G.y = None

        print(G)
        return G

    def getValidMoves(self, path):
        """
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current state,
                        0 for invalid moves
        """

        valid_moves = [1 for _ in range(self.n)]
        for node in path:
            valid_moves[node] = 0

        return valid_moves
        

    def path_pay(self, path):
        """
        path: a tuple of indices between 1 and self.args.n_nodes-1
        """
        complete_path = [0] + list(path) + [0]
        pay = 0
        for i in range(len(complete_path) - 1):
            u = complete_path[i]
            v = complete_path[i+1]
            pay += self.graph[u][v]['weight']
        return pay

    def optimal_solution(self):
        
        optimal_path = None
        optimal_cost = float('inf')

        # brute force
        path = np.arange(self.n)[1:]
        for permutation in permutations(path):
            cost = self.path_pay(permutation)

            if cost < optimal_cost:
                optimal_cost = cost
                optimal_path = permutation

        return (optimal_cost, optimal_path)

