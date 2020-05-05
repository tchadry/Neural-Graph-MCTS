from TspNN import NNetWrapper as nn
from util import dotdict
from TspGame import TspGame
import numpy as np

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
    'n_nodes': 8,
    'n_node_features': 5,
    'n_executions': 100,

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'use_gdc': True

})

#sims = [5, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
#sims = [250]
#sims = [250, 800, 900, 1000]

n_games = 300
error = 1.4

net = nn(args)

#ims = [5, 25, 50]
files = ['best_model_25_mctsSims.pth.tar', 'best_model_50_mctsSims.pth.tar', 'best_model_5_mctsSims.pth.tar',
         'checkpoint_5sims_1.pth.tar', 'best_model_75_mctsSims.pth.tar', 'checkpoint_25sims_3.pth.tar',
         'checkpoint_50sims_1.pth.tar']

for file in files:
    #file = f'best_model_{sim}_mctsSims.pth.tar'
    args.checkpoint = './new_models/'
    #file = 'best.pth.tar'
    net.load_checkpoint(args.checkpoint, filename=file)
    wins = 0

    for i in range(n_games):

        #if ((i + 1) % 5) == 0:
        #   print(f'Playing game {i+1}.')

        game = TspGame(args.n_nodes)
        path = game.getInitBoard()

        while not game.getGameEnded(path):

            probabilities = net.predict(game.construct_graph(path))[0]
            probabilities = np.array([1/(x+1) for x in probabilities])
            valid_moves = game.getValidMoves(path)
            valids = valid_moves*probabilities

            move = None
            if sum(valids) == 0:
                move = np.argmax(valid_moves)
            else:
                move = np.argmax(valid_moves)
            path.append(move)

        path = path[1:]

        #print(path)
        #input()
        path_cost = game.path_pay(path)
        optimal_cost = game.optimal_solution()[0]

        if optimal_cost*error >= path_cost:
            wins += 1

    print(f'{file}: {wins / n_games}')