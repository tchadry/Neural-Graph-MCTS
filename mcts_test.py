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

    mcts = MCTS(game, None, args

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



games, optimal = create_mcts_games(20, 10)



num_simulations = [1000,1500,2000]

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
