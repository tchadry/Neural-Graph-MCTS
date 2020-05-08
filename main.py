import torch
#from TspNN import  NNetWrapper
from TspNN import NNetWrapper as nn
import os
from TspGame import TspGame
from Coach import Coach


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


#arguments and their explanations

args = dotdict({

    'tempThreshold': 15,        #
    'updateThreshold': 0.55,    #During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate - the MCTS iterations
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './gcn/',
    'load_model': False,
    'load_folder_file': ('./temp2/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,




    'n_node_features': 5, #number of features in the node representations for GNN
    'n_executions': 100, #how many times we will be running our main loop and saving the iterations:


    'lr': 0.001,


    'batch_size': 64,
    'use_gdc': True,


    'invert_probs': False,
    'n_nodes': 8, #number of nodes in the TSP problem
    #'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'device': 'cpu',
    'cuda': False,

    'numIters': 50,  # number of times checkpoint will be saved during coach
    'numEps': 50,  # Number of complete self-play games to simulate during a new iteration.
    'dropout': 0.3,
    'epochs': 10,


})
#args = dotdict({
    #'lr': 0.001,
    #'dropout': 0.3,
    #'epochs': 10,
    #'batch_size': 64,
    #'num_channels': 512,
#})

nnet = nn(args)

if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])


#mcts_sims = num_simulations = [5, 25, 50, 100, 200, 400, 1000]#, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000]

mcts_sims = [100]#, 100, 300, 500, 1000, 2000]

for n_sim in mcts_sims:

    args.numMCTSSims = n_sim

    g = TspGame(args.n_nodes)
    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()


    c.nnet.save_checkpoint(folder=args.checkpoint, filename=f'best_model_{args.numMCTSSims}_mctsSims.pth.tar')

