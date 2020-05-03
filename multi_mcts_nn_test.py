from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt
from TspGame import TspGame
from util import dotdict
from TspNN import NNetWrapper as nn
from util import dotdict
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess



initial_args = dotdict({
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


def predict_path(game, simulations, args, nnet=None):
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


n_nodes = initial_args.n_nodes

results = {}

##########################################################################################################
# Last level parallelization

global_net = None
global_args = None
global_sim = None

def run_game(i):

    predicted_cost = predict_path(games[i], global_sim, global_args, global_net)

    return optimal[i] * error >= predicted_cost



def last_test_model(file):
    print(f'File: {file}')

    global global_args
    global global_net
    global global_sim

    global_args = initial_args
    global_args.checkpoint = './temp_models/'
    global_net = nn(global_args)

    result = []

    if file != 'none':
        global_net.load_checkpoint(global_args.checkpoint, filename=file)
    else:
        global_net = None

    for sim in num_simulations:

        #print("testing simulations", sim)

        global_sim = sim

        pool = Pool()
        game_results = pool.map(run_game, list(range(n_games)))
        pool.close()
        pool.join()



        result.append(sum(game_results) / n_games)

    #print(result)

    return result

############################################################################################################

def test_model(file):
    args = initial_args
    print(f'File: {file}')
    print('################################################################################')

    result = []

    args.checkpoint = './temp_models/'
    net = nn(args)

    if file != 'none':
        net.load_checkpoint(args.checkpoint, filename=file)
    else:
        net = None

    for sim in num_simulations:

        print("testing simulations", sim)
        wins = 0

        for i in range(n_games):
            predicted_cost = predict_path(games[i], sim, args, net)

            if optimal[i] * error >= predicted_cost:
                wins += 1

        result.append(wins / n_games)
        # print(result)

    #print(result)

    return (file, result)

#######################################################################################################################
# Multiprocessing for num_simulations

def run_simulation(arguments):

    sim, args, file = arguments

    args.numMCTSSims = sim

    wins = 0

    # Load net
    net = nn(args)
    if file != 'none':
        net.load_checkpoint(args.checkpoint, filename=file)
    else:
        net = None

    # Simulate n games
    for i in range(n_games):
        predicted_cost = predict_path(games[i], sim, args, net)

        if optimal[i] * error >= predicted_cost:
            wins += 1

    return (sim, wins / n_games)


#global_list = []

def multi_test_model(file):
    #print(file, sim_pools)


    args = initial_args

    args.checkpoint = './temp_models/'


    new_pool = Pool(sim_pools)

    tuple_results = new_pool.map(run_simulation, list(zip(num_simulations, [args]*len(num_simulations), [file]*len(num_simulations))))

    #tuple_results = new_pool.map(run_simulation, global_list)


    # Sort by number of simulations performed
    tuple_results = sorted(tuple_results, key=lambda x: x[0])
    result = [x[1] for x in tuple_results]

    return (file, result)
#######################################################################################################################
n_games = 100
error = 1.01

games, optimal = create_mcts_games(n_games, n_nodes)


num_simulations = [5, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]#, 600, 700, 800]#, 900, 1000, 1500, 2000]#, 2500]

# NB: include 'none' as a file for mcts only evaluaton
files = ['inv0_1.tar', 'inv0_2.tar', 'none']

print(f'games: {n_games}\n'
      f'error: {error}\n'
      f'nodes: {initial_args.n_nodes}')




parallel_models = True
parallel_simulations = True
last_level = True
cpu_count = multiprocessing.cpu_count()
pools = len(files)
sim_pools = (cpu_count - pools) // pools


if last_level:
    for file in files:
        result = last_test_model(file)
        results[file] = result
elif parallel_models:
    # Only assign each cpu to a pool
    if cpu_count < 3*len(files) or not parallel_simulations:
        pool = Pool()
        tuple_results = pool.map(test_model, files)
        for t in tuple_results:
            results[t[0]] = t[1]

        pool.close()
        pool.join()

    else:
        pool = MyPool(pools)
        tuple_results = pool.map(multi_test_model, files)
        for t in tuple_results:
            results[t[0]] = t[1]
        pool.close()
        pool.join()
else:
    for file in files:
        result = test_model(file)
        results[result[0]] = result[1]



print(results)
