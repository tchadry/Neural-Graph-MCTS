import _pickle as pickle
import torch
from TspGame import TspGame


def create_mcts_games(num_games, nodes):
    """
    returns list of games and list of optimal game values
    """

    games = []
    optimal_results = []

    for i in range(num_games):
        print(i)
        print(i)
        current = TspGame(nodes)
        optimal_res, _ = current.optimal_solution()
        games.append(current)
        optimal_results.append(optimal_res)
        
    return games, optimal_results


games, optimal= create_mcts_games(num_games=20, nodes=10)

games_10_nodes={}

for i,j in games,optimal:
    games_10_nodes[i]=j


with open('10nodestest.txt', 'w') as file:
     file.write(pickle.dumps(games_10_nodes))
