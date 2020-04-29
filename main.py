import torch
#from TspNN import  NNetWrapper
from old_wrapper import NNetWrapper as nn
import os
from TspGame import TspGame
from Coach import Coach


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'numIters': 10,
    'numEps': 50,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,    #During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'n_nodes': 8,
    'n_node_features': 5,
    'n_executions': 100
})

nnet = nn(args.n_node_features, args.n_nodes, True)

if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

for i in range(0, args.n_executions):
    print(f'Execution nr. {i + 1}')

    g = TspGame(args.n_nodes)
    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

    c.nnet.save_checkpoint(folder=args.checkpoint, filename=f'Iteration_{i+1}')