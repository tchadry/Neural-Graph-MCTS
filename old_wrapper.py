import os
import shutil
import time
import random
import numpy as np
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from util import dotdict, AverageMeter
from GNN import GNN as Net
import datetime

N_NODES = 8

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'use_gdc': True,
    'n_node_features': 5,
    'n_nodes': N_NODES,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_nodes_nnet': N_NODES,
})

class NNetWrapper():
    def __init__(self, args):
        self.nnet =  Net(args)
        self.action_size = args.n_nodes

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        print("- training model with {} examples".format(len(examples)))

        for epoch in range(args.epochs):
            if epoch in [0, 4, 9]:
                print('EPOCH ::: ' + str(epoch+1), datetime.datetime.now())
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            # bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)

                # boards: list of pytorch geometric Data objects
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    target_pis, target_vs = target_pis.contiguous().cuda(), target_vs.contiguous().cuda()


                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi = torch.zeros_like(target_pis)
                out_v = torch.zeros_like(target_vs)
                for idx, board in enumerate(boards):
                    sample_pi = pis[idx]
                    sample_v = vs[idx]
                    board = board.to(args.device)
                    pred_pi, pred_v = self.nnet(board)  # board: pytorch geometric Data object
                    out_pi[idx] = pred_pi
                    out_v[idx] = pred_v
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(),len(boards))
                v_losses.update(l_v.item(), len(boards))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1


    def predict(self, board):
        """
        board: pytorch geometric Data object
        """
        # timing
        start = time.time()

        # preparing input
        if args.cuda: board = board.to(args.device)
        self.nnet.eval()
        with torch.no_grad():
            pred = self.nnet(board)
            if isinstance(pred, tuple):
                pi, v = pred
            else:
                pi = None
                v = pred
#             pi, v = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        if pi is None:
            return pi, v.item()
        return pi.data.cpu().numpy(), v.item()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
