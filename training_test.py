from util import dotdict
from TspGame import TspGame
from MCTS import MCTS
import numpy as np
from random import shuffle
import datetime
# train the model: check training examples
# training example: (current graph, pi, v)
# pi is the MCTS informed policy vector, v is +1 if
# the player eventually won the game, else -1.

N = 8

args = dotdict({
    'numMCTSSims': 200,
    'n_nodes': N,
    'cpuct': 1,
    'use_gdc': False,
    'n_node_features': 3
    })

training_examples = []
num_games = 500
for i in range(num_games):
    total_pay = 0
    current_training_examples = []
    game = TspGame(N)
    mcts = MCTS(game, None, args)
    path = [0] # starting positions, path is only at 0

    while not game.getGameEnded(path):
        pi = mcts.getActionProb(path)
        action = np.argmax(pi)
        print(pi)
        current_training_examples.append([game.construct_graph(path), pi])
        path, pay = game.getNextState(path, action)
        total_pay += pay
        
        if game.getGameEnded(path):
            for example in training_examples:
                example.append(pay)
            training_examples = [tuple(ex) for ex in training_examples]

    training_examples = training_examples + training_examples

    if i % 20 == 0:
        print("------> Trained:   " + str(len(training_examples)) ) 