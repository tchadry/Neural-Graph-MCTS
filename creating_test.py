import _pickle as pickle
import torch
from TspGame import TspGame


def create_mcts_games(num_games, nodes):
    """
    returns list of games and list of optimal game values
    """

    games_10_nodes={}
    optimal_results=[]

    for i in range(num_games):
        print(i)
        print(i)
        current = TspGame(nodes)
        optimal_res, _ = current.optimal_solution()
        games_10_nodes[current]=optimal_res
        optimal_results.append(optimal_res)
        
    return games_10_nodes, optimal_results


games, optimal= create_mcts_games(num_games=20, nodes=11)




with open('11nodestestfile2', 'wb') as file:
     pickle.dump(games,file)
