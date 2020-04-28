import math
import numpy as np
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, path, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(path)

        current_path = tuple(path)
        counts = [self.Nsa[(current_path, a)] if (
            current_path, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        print(counts)
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        print(counts_sum)
        probs = [x/counts_sum for x in counts]
        return probs

    def search(self, path):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        current = path
        current_path = tuple(path)
        n = self.game.getActionSize()

        # if not in self.Es, we include with binary of whether terminal state
        if current_path not in self.Es:
            # getgameended returns 1 if terminal, 0 otherwise
            self.Es[current_path] = self.game.getGameEnded(current)

        # then,we check to see if it is terminal node
        if self.Es[current_path] != 0:
            return 0

        # then, we check if in self.PS, which stores initial policy
        if current_path not in self.Ps:
            # leaf node
            #self.Ps[s], v = self.nnet.predict(canonicalBoard)

            # if we are testing with GNN
            if self.nnet is not None:
                pred_p, v = self.nnet.predict(
                    self.game.construct_graph(current))
                # check if were predicting both
                if pred_p is not None:
                    # assign policy vector to pred_p
                    self.Ps[current_path] = pred_p
                else:
                    self.Ps[current_path] = np.ones(n)

            # if testing only with MCTS
            else:
                v = 0
                self.Ps[current_path] = np.ones(n)

            # now, we check for all valid moves
            valids = self.game.getValidMoves(current)

            # only keep the policy values for valid moves
            self.Ps[current_path] = self.Ps[current_path]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[current_path])
            if sum_Ps_s > 0:
                self.Ps[current_path] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[current_path] = self.Ps[current_path] + valids
                self.Ps[current_path] /= np.sum(self.Ps[current_path])

            self.Vs[current_path] = valids
            self.Ns[current_path] = 0
            return -v

        valids = self.Vs[current_path]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        print(current)
        print('evaluating valid actions in search')
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # compute ucb
                if (current_path, a) in self.Qsa:
                    u = self.Qsa[(current_path, a)] + self.args.cpuct*self.Ps[current_path][a] * \
                        math.sqrt(self.Ns[current_path]) / \
                        (1+self.Nsa[(current_path, a)])
                else:
                    u = self.args.cpuct * \
                        self.Ps[current_path][a] * \
                        math.sqrt(self.Ns[current_path] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_path, pay = self.game.getNextState(current, a)

        v = self.search(next_path) + pay

        if (current_path, a) in self.Qsa:
            self.Qsa[(current_path, a)] = (self.Nsa[(current_path, a)] *
                                           self.Qsa[(current_path, a)] + v)/(self.Nsa[(current_path, a)]+1)
            self.Nsa[(current_path, a)] += 1

        else:
            self.Qsa[(current_path, a)] = v
            self.Nsa[(current_path, a)] = 1

        self.Ns[current_path] += 1
        return -v
