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
    'numIters': 3, #number of times checkpoint will be saved during coach 
    'numEps': 50,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,    #During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate - the MCTS iterations
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'cuda': False,

    'checkpoint': './10nodesnew/',
    'load_model': False,
    'load_folder_file': ('./10nodesnew/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'n_nodes': 10, #number of nodes in the TSP problem 
    'n_node_features': 5, #number of features in the node representations for GNN 
    'n_executions': 20, #how many times we will be running our main loop and saving the iterations: 

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'use_gdc': True
    
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

for i in range(0, args.n_executions):
    print(f'Execution nr. {i + 1}')

    g = TspGame(args.n_nodes)
    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

    c.nnet.save_checkpoint(folder=args.checkpoint, filename=f'Iteration_{i+1}')

    #change test - edit
    #edit 
    #unclear whether we should test withcheckpoint or with iterations or ith best model. for now test with all of them 
