import errno
import os
import sys
import time
import matplotlib.pyplot as plt
import math
from TspGame import TspGame
from util import dotdict
import numpy as np
from MCTS import MCTS
import datetime
from TspNN import NNetWrapper
import torch



def create_mcts_games(num_games, nodes):
    """
    returns list of games and list of optimal game values
    """

    games = []
    optimal_results = []

    for i in range(num_games):
        print(i)
        current = TspGame(nodes)
        optimal_res, _ = current.optimal_solution()
        games.append(current)
        optimal_results.append(optimal_res)
        
    return games, optimal_results


def run_simulations(game, simulations, nodes, model=None):
    """
    Given a TSP problem, run MCTS with `n_simulations`
    simulations before choosing an action.

    Returns value of MCTS path
    """

    args = dotdict({
        'numMCTSSims': simulations,
        'n_nodes': nodes,
        'cpuct': 1
        })

    mcts = MCTS(game, model, args)

    current = [0]
    pay = 0
    decisions = []

    end = game.getGameEnded(current)

    while not end:
        action = np.argmax(mcts.getActionProb(current))
        current, payoff = game.getNextState(current, action)
        decisions.append(action)
        pay += payoff
        end = game.getGameEnded(current)

    return  game.path_pay(tuple(decisions))


n_games = 20
n_nodes = 5
games, optimal= create_mcts_games(num_games=n_games, nodes=n_nodes)
n_simulations = 500


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'use_gdc': True,
    'n_node_features': 5,
    'n_nodes': n_nodes,
    'n_nodes_nnet': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'numMCTSSims': n_simulations,
    'max_dist': 100,
    'cpuct': 1,
})

nnet = NNetWrapper(args)

results_no = []
results_nn = []
checkpoints_nn=[]


folder='./temp4/'
filename='best.pth.tar'
nnet.load_checkpoint(folder, filename)

print("Beginning simulations -", datetime.datetime.now())
start = datetime.datetime.now()
for sims in np.arange(100,1000,100):
    print("Beginning game checkpoint:", cp, datetime.datetime.now())


    good = 0
    for i in range(n_games):
        print(i)

        score_of_mcts_path = run_simulations(games[i], sims, nodes=n_nodes, model=nnet)

        if score_of_mcts_path/optimal[i] < 1.10:
            good += 1

    checkpoints_nn.append(sims)
    results_nn.append(good/n_games)
    print("results_nn:  " + str(results_nn))
        
print("DONE.", datetime.datetime.now())
end = datetime.datetime.now()
print("Elapsed:", end - start)

for sims in np.arange(100,1000,100):
    good = 0
    for i in range(n_games):
        if i % 10 == 0: print("- Game", i, datetime.datetime.now())
        args = dotdict({
            'numMCTSSims': sims,
            'n_nodes': N_NODES,
            'max_dist': 100,
            'cpuct': 1,
            'max_dist': 100
            })

        mcts = MCTS(games[i], None, args)

        state = [0]
        R = 0
        Actions = []

        while not games[i].getGameEnded(state):
            action = np.argmax(mcts.getActionProb(state))
            state, reward = games[i].getNextState(state, action)
            Actions.append(action)
            R += reward
            
        score_of_mcts_path = games[i].path_pay(Actions)

        if score_of_mcts_path/optimal[i] < 1.1:
            good+= 1


    results_no.append(good/n_games)

print(results_no)
print(results_nn)


#x = []
#y = []
#for key in sorted(results_nn):
    #x.append((key+1)*10)
  # y.append(results_nn[key])

nnet_100_line = plt.plot(checkpoints_nn, results_nn, label='MCTS with NNet, 100 sims')
mcts_100_line = plt.plot(checkpoints_nn, results_no,'MCTS without NNet, 100 sims')
plt.xlabel('Self-Play Epochs with NNet')
plt.ylabel('Percent Games within 1.1 of Optimal')
plt.legend()
plt.show()
