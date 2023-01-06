import pandas as pd
from torch.utils import data
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='ResNet')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                            help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                            help='input batch size for training (default: 4)')
    #parser.add_argument('--shuffle', type=bool, default=True, metavar='SF',
    #                        help='shuffle for training (default: True)')
    #parser.add_argument('--test-shuffle', type=bool, default=False, metavar='SF',
    #                        help='shuffle for testing (default: False)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='learning rate (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
    # must add args=[] in notebook version
    args = parser.parse_args(args=[])

    return args
