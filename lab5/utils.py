import os
import torch
import torch.utils.data as data
from __future__ import unicode_literals, print_function, division
import random
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import copy

def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--SOS_token', type=int, default=0, metavar='N',
                            help='start of sentence token (default: 0)')
    parser.add_argument('--EOS_token', type=int, default=1, metavar='N',
                            help='end of sentence token (default: 1)')
    parser.add_argument('--input_size', type=int, default=28, metavar='N',
                            help='the number of vocabulary {SOS, EOS, a,..., z} (default: 28)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                            help='LSTM hidden layer size (default: 256)')
    parser.add_argument('--conditional_size', type=int, default=8, metavar='N',
                            help=' size of conditional data (default: 8)')
    parser.add_argument('--LR', type=float, default=0.05, metavar='N',
                            help='learning rate (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=500, metavat='N',
                            help='learning epochs (default: 500)')
    parser.add_argument('--kl_annealing_type', type=str, default='cycle', metavar='S',
                            help='KL annealing method (default: cycle)')
    parser.add_argument('--time', type=int, default=2, metavar='N',
                            help='threshold of KL weights (default: 2)')
    parser.add_argument('--max_length', type=int, default=15, metavar='N',
                            help='maximum decoder input size (default: 15)')
    parser.add_argument('--file_path', type=str, default='best_monotonic_time1000_epochs500.pt', metavar='S',
                            help='file path of model weight (default: best_monotonic_time1000_epochs500.pt)')
    parser.add_argument('--test_time', type=int, default=20, metavar='N',
                            help='test time of BLEU score and Gaussian score (default: 20)')
    # must add args=[] in notebook version
    args = parser.parse_args(args=[])

    return args

def compute_bleu(output, reference):
    """
    :param output: The word generated by your model
    :param reference: The target word
    :return:
    """
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output,weights=weights, smoothing_function=cc.method1)

def loss_function(predict_distribution, predict_output_length, target, mu, logvar):
    """
    :param predict_distribution: (target_length,28) tensor
    :param predict_output_length:           (may contain EOS)
    :param target: (target_length) tensor   (contain EOS)
    :param mu: mean in VAE
    :param logvar: logvariance in VAE
    :return: CrossEntropy loss, KL divergence loss
    """
    Criterion=nn.CrossEntropyLoss()
    CEloss=Criterion(predict_distribution[:predict_output_length],target[:predict_output_length])

    # KL(N(mu,variance)||N(0,1))
    KLloss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return CEloss, KLloss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_teacher_forcing_ratio(epoch, epochs):
    """
    :param epoch: current epoch number
    :param epochs: epoch size
    """
    # 1, 1-1.0/epoch_size, 1-1.0/epoch_size-1.0/epoch_size, ...
    # decrease 1.0/epoch_size in every epoch
    teacher_forcing_ratio = 1.-(1./(epochs-1))*(epoch-1)
    return teacher_forcing_ratio

def get_kl_weight(epoch,epochs,kl_annealing_type,time):
    """
    :param epoch: current epoch number
    :param epochs: epoch size
    :param kl_annealing_type: KL annealing method
    :param time: threshold of KL weights
    """
    # kl_annealing_type: "monotonic"/"cycle"
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle', 'kl_annealing_type not exist!'

    # time: theshold to control when to become 1
    # increase 1.0/epoch_size in every epoch 
    if kl_annealing_type == 'monotonic':
        return (1./(time-1))*(epoch-1) if epoch<time else 1.

    else: #cycle
        period = epochs//time
        epoch %= period
        KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
        return KL_weight

def get_gaussian_score(words):
    """
    :param words:
    words = [['consult', 'consults', 'consulting', 'consulted'],
             ['plead', 'pleads', 'pleading', 'pleaded'],
             ['explain', 'explains', 'explaining', 'explained'],
             ['amuse', 'amuses', 'amusing', 'amused'], ....]
    """
    words_list = []
    score = 0
    yourpath = os.path.join('lab5_data','train.txt')  #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list):
    """
    :param epochs: epoch size
    :param CEloss_list: 
    :param KLloss_list: from train
    :param BLEUscore_list: from evaluate
    :param teacher_forcing_ratio_list:
    :param kl_weight_list: from user setting
    """
    x=range(1,epochs+1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,CEloss_list, label='CEloss')
    plt.plot(x,KLloss_list, label='KLloss')

    plt.plot(x,BLEUscore_list,label='BLEU score')

    plt.plot(x,teacher_forcing_ratio_list,linestyle=':',label='tf_ratio')
    plt.plot(x,kl_weight_list,linestyle=':',label='kl_weight')
    plt.legend()

    return fig