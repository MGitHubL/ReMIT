import sys
import math
import torch
import ctypes
import datetime
import numpy as np
import argparse
import time
import random
import os

from model import ReMIT
from experiments import experiment

FType = torch.FloatTensor
LType = torch.LongTensor

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

def main_train(args):
    start = datetime.datetime.now()
    the_train = ReMIT.ReMIT(args)
    the_train.train()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))


def main_test(args, task, method):
    the_test = experiment.Exp(args.dataset, method)
    if task == 'nc':
        print('Node Classification in ' + str(args.dataset) + ' of ' + str(method) + '.')
        the_test.nc()
    if task == 'lp':
        print('Link Prediction in ' + str(args.dataset) + ' of ' + str(method) + '.')
        the_test.lp()
    if task == 'nv':
        print('Network Visualization in dblp' + ' of ' + str(method) + '.')
        the_test.nv()


if __name__ == '__main__':

    data = 'bitalpha'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--cluster', type=int, default=5)
    parser.add_argument('--hist_len', type=int, default=1)
    parser.add_argument('--neg_size', type=int, default=3)
    parser.add_argument('--lambdas', type=int, default=0.01)

    parser.add_argument('--batch_size', type=int, default=1024)
    # [b:lr]=1024:0.01
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    args = parser.parse_args()

    print(args)
    main_train(args)

    # main_test(args, task='lp', method='ReMIT_20')
    main_test(args, task='nc', method='ReMIT_10')
    '''
    task = [nc:node classification, lp:link prediction, nv:network visualization]
    dataset of nctask = [dblp, bitotc, bitalpha], dataset of lp = all, dataset of nv = [dblp]
    method = [GET_epoch, htne_epoch, deepwalk, nodevec]
    _epoch is the number of epoch, such as 'HT_50'
    '''