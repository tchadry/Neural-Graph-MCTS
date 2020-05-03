from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt
from TspGame import TspGame
from util import dotdict
from TspNN import NNetWrapper as nn
from util import dotdict
from main import args


def create_mcts_games(num_games, nodes=8):
    """
    returns list of games and list of optimal game values
    """

    games = []
    optimal_results = []

    for i in range(num_games):

        current = TspGame(nodes)
        optimal_res, _ = current.optimal_solution()
        games.append(current)
        optimal_results.append(optimal_res)
        
    return games, optimal_results


def predict_path(game, simulations, nnet=None):
    """
    Given a TSP problem, run MCTS with `n_simulations`
    simulations before choosing an action.

    Returns value of MCTS path
    """

    args.numMCTSSims = simulations

    mcts = MCTS(game, nnet, args)

    path = [0]

    while not game.getGameEnded(path):
        predictions = mcts.getActionProb(path)
        action = np.argmax(predictions)
        path.append(action)

    return  game.path_pay(tuple(path))


n_games = 10
n_nodes = args.n_nodes

games, optimal = create_mcts_games(n_games, n_nodes)

# temp4 chkpnt1, 8 nodes
# [0.01, 0.01, 0.03, 0.05, 0.07, 0.13, 0.14, 0.11, 0.16, 0.21, 0.32, 0.29, 0.41, 0.46, 0.5, 0.43, 0.53, 0.71, 0.69, 0.78, 0.69, 0.74, 0.85, 0.77]

num_simulations = [500,1000,1500,2000]# 1500, 2000, 2500]

files = ['best.pth.tar', 'none']


for file in files:
    print(f'File: {file}')
    print('################################################################################')

    result = []

    args.checkpoint = './10nodesFINAL/'
    net = nn(args)

    if file != 'none':
        net.load_checkpoint(args.checkpoint, filename=file)
    else: 
        net = None

    for sim in num_simulations:

        print("testing simulations", sim)
        wins = 0

        for i in range(n_games):
            predicted_cost = predict_path(games[i],  sim, net)

            if optimal[i]*1.1 >= predicted_cost:
                wins += 1

        result.append(wins/n_games)
    print(file)
    print(result)

    print("Simulations done")
    plt.title("Plotting MCTS percentage within 1.1 of optimal with change in iterations")
    plt.xlabel('Number ofsimulations')
    plt.ylabel('MCTS Results')
    plt.plot(num_simulations, result)
    plt.show()
