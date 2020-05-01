from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt
from TspGame import TspGame
from util import dotdict
from TspNN import NNetWrapper as nn
from util import dotdict

args = dotdict({
    'numIters': 10,
    'numEps': 50,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.55,
# During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 50,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'cuda': False,

    'checkpoint': './temp2/',
    'load_model': False,
    'load_folder_file': ('./temp2/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    # 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'device': 'cpu',
    'n_nodes': 25,
    'n_node_features': 5,
    'n_executions': 100,

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'use_gdc': True,
    'invert_probs': True


})

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

    #print(path)
    #input()
    return  game.path_pay(tuple(path[1:]))



n_games = 100
error = 1.01
n_nodes = args.n_nodes

games, optimal = create_mcts_games(n_games, n_nodes)

# temp4 chkpnt1, 8 nodes
# [0.01, 0.01, 0.03, 0.05, 0.07, 0.13, 0.14, 0.11, 0.16, 0.21, 0.32, 0.29, 0.41, 0.46, 0.5, 0.43, 0.53, 0.71, 0.69, 0.78, 0.69, 0.74, 0.85, 0.77]

num_simulations = [5, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500]

#files = ['best.pth.tar', 'checkpoint_1.pth.tar', 'checkpoint_2.pth.tar', 'checkpoint_4.pth.tar', 'checkpoint_8.pth.tar', 'none']
#files = ['checkpoint_1.pth.tar', 'none', 'checkpoint_50sims_1.pth.tar']
files = ['50sims_25nodes.pth.rar', 'none']

print(f'games: {n_games}\n'
      f'error: {error}\n'
      f'nodes: {args.n_nodes}')

for file in files:
    print(f'File: {file}')
    print('################################################################################')

    result = []

    args.checkpoint = './temp4/'
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

            if optimal[i]*error >= predicted_cost:
                wins += 1

        result.append(wins/n_games)
        print(result)

    #print("Simulations done")
    #print("Hello")
    #plt.title("Plotting MCTS percentage within 1.1 of optimal with change in iterations")
    #plt.xlabel('Number ofsimulations')
    #plt.ylabel('MCTS Results')
    #plt.plot(num_simulations, result)
    #plt.show()
