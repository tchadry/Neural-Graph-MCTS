"""NNet- inspired from alphazero Othello"""

from util import dotdict, AverageMeter
import tensorflow as tf
import torch
import os
import shutil
import time
import random
import numpy as np
import math
import sys
from GNN import GNN
sys.path.append('../../')


#args = dotdict({
    #'lr': 0.001,
    #'dropout': 0.3,
    #'epochs': 10,
    #'batch_size': 64,
    #'num_channels': 512,
#})

# we are gonna gave to pass d, num_nodes and use_gdc to the neural net, or define those
#LOOK AT HOW WE ARE GETTING THE ARGS AND PUSHING IT TO NNET

class NNetWrapper():
    def __init__(self, args):
        self.nnet = GNN(args)
        self.action_size = args.n_nodes

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        print('training')
        optimizer = torch.optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))

            # first train
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < len(examples)//args.batch_size:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # get target data
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                final_pis = torch.FloatTensor(np.array(pis))
                final_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                for idx, board in enumerate(boards):
                    board = board.to(args.device)
                    pred_pi, pred_v = self.nnet(board)
                    final_pis[idx] = pred_pi
                    final_vs[idx] = pred_v

                # compute and record losses - we need to create these loss functions
                pi_loss = self.pi_loss(target_pis, final_pis)
                v_loss = self.v_loss(target_vs, final_vs)

                total_loss = pi_loss + v_loss

                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                # optimizer steps
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # add prints to track progress

    def pi_loss(self, targets, outputs):
        # fill in
        return -torch.sum(targets*outputs)/targets.size()[0]

    def v_loss(self, targets, outputs):
        # fill in
        res = (targets-outputs.view(-1))**2
        return torch.sum(res)/targets.size()[0]

    def predict(self, board):

        # do predict
        """
        board: pytorch object
        """

        # preparing input
        self.nnet.eval()

        with torch.no_grad():
            pred = self.nnet(board)
            if isinstance(pred, tuple):
                pi, v = pred
            else:
                pi = None
                v = pred


        if pi is not None:
            return pi.data.cpu().numpy(), v.item()
        else:
            return pi, v.item()


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            raise Exception("No model in path {}".format(filepath))
        self.nnet.load_state_dict(torch.load(filepath))
