from util import dotdict

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

n_games_to_train = 1000

for i in range(n_games_to_train):
    R = 0
    game_examples = []
    game = TspGame(args)
    mcts = MCTS(game, None, args)

    state = game.getInitState()
    while not game.getGameEnded(state):
        pi = mcts.getActionProb(state)
        game_examples.append([game.build_graph_from_state(state), pi])
        action = np.argmax(pi)
        state, reward = game.getNextState(state, action)
        R += reward
        if game.getGameEnded(state):
            for example in game_examples:
                example.append(R)
            game_examples = [tuple(x) for x in game_examples]
    training_examples.extend(game_examples)
    if i % 50 == 0:
        print("- Completed game {} -- {} training example done".format(i, len(training_examples)), datetime.datetime.now())