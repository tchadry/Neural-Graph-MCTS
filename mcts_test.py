from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt
from TspGame import TspGame
from util import dotdict


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


def run_simulations(game, simulations, nodes):
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

    mcts = MCTS(game, None, args)

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

    print('decisions:', decisions)
    return  game.path_pay(tuple(decisions))



n_games = 100
n_nodes = 20



# results for 100 x 8 games
# [0.0, 0.21, 0.36, 0.47, 0.5, 0.54, 0.62, 0.65, 0.67, 0.68, 0.71, 0.72, 0.76, 0.74, 0.75, 0.78, 0.83, 0.86, 0.93, 0.93, 0.94, 0.97, 0.97, 0.99]

# results for 100 x 9
# [0.01, 0.14, 0.26, 0.32, 0.28, 0.36, 0.45, 0.49, 0.5, 0.53, 0.58, 0.56, 0.62, 0.69, 0.71, 0.73, 0.75, 0.8, 0.82, 0.86, 0.86, 0.85, 0.87, 0.91]

num_simulations = [5, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]#, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000]

result = []


for sim in num_simulations:
    print("testing simulations", sim)

        
    good = 0

    for i in range(20):
        mcts_res = run_simulations(games[i],  sim, 10)
            
        if mcts_res/optimal[i] < 1.1:
            good += 1

    result.append(good/20)
    print(result)

print("Simulations done")
print("Hello")
plt.title("Plotting MCTS percentage within 1.1 of optimal with change in iterations")
plt.xlabel('Number ofsimulations')
plt.ylabel('MCTS Results')
plt.plot(num_simulations, result)
plt.show()
