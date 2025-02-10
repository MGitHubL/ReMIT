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

    data = 'ml1m'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    args = parser.parse_args()

    print(args)

    main_test(args, task='nc', method='S2T')

    '''
    task = [nc:node classification, lp:link prediction, nv:network visualization]
    dataset of nctask = [dblp, bitotc, bitalpha], dataset of lp = all, dataset of nv = [dblp]
    method = [GET_epoch, htne_epoch, deepwalk, nodevec]
    _epoch is the number of epoch, such as 'HT_50'
    '''
