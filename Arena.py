import numpy as np
from util import AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):

        #edit this 
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """


        #we are going to let the first player play all their actions first until
        # a terminal state, then the second player. Then we compare the results
        #for both: 

        #get initial state 
        current = self.game.getInitState() 

        #defining the needed variables
        first_pay=0
        first_actions=[]
        second_pay=0
        second_actions=[]

        #player 1
        end=self.game.getGameEnded(current)
        while not end:
          #we play until it reaches a terminal state
          decision = self.player1(current)

          #define next state and pay
          current,pay = self.game.getNextState(current,decision)

          #update pay
          first_pay+=pay

          #update actions 
          first_actions.append(decision)

          end=self.game.getGameEnded(current)


        #player 2
        current=self.game.getInitState()
        end=self.game.getGameEnded(current)
        while not end:
          #we play until it reaches a terminal state
          decision = self.player2(current)

          #define next state and pay
          current,pay = self.game.getNextState(current,decision)

          #update pay
          second_pay+=pay

          #update actions 
          second_actions.append(decision)

          end=self.game.getGameEnded(current)


        if first_pay > second_pay:
          return 1
        elif second_pay>first_pay:
          return -1
        else:
          return 0
          
    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        #bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        #why num*2
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()

        return oneWon, twoWon, draws