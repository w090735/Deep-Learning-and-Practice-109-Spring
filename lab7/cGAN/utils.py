import torch
from torch.utils.data import Dataset

import json
import numpy as np
from PIL import Image

import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import pickle

import torchvision.models as models

def set_args():
	parser = argparse.ArgumentParser(description='cGAN')
	parser.add_argument('--num_epochs', type=int, default=400, metavar='N',
                            help='epoch number (default: 400)')
	parser.add_argument('--batch_size', type=int, default=128, metavar='N',
							help='batch size of train data (default: 128)')
	parser.add_argument('--lr_D', type=float, default=0.0002, metavar='F',
							help='learning rate of discriminator (default: 0.0002)')
	parser.add_argument('--lr_G', type=float, default=0.0002, metavar='F',
							help='learning rate of generator (default: 0.0002)')
	parser.add_argument('--latent_size', type=int, default=104, metavar='N',
							help='latent size of generator input (default: 104)')
	parser.add_argument('--image_size', type=int, default=128, metavar='N',
							help='image size of train data (default: 128)')
	parser.add_argument('--d_aux_weight', type=int, default=48, metavar='N',
							help='fixed weight (default: 48)')
	parser.add_argument('--path', type=str, default='/home/auser03/lab7/task_1/', metavar='S',
							help='path of dataset (default: /home/auser03/lab7/task_1/)')
	args = parser.parse_args(args=[])

	return args

# Weights
# initialize model weight
def weights_init(m):
    classname = m.__class__.__name__

    # conv: N(0, 0.02)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # batchNorm: N(1, 0.02) + 0 
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# update generator aux weight
# linear increase from 0 to 24 in first 1000 batch
def g_aux_weight(iter):
    max_w = d_aux_weight / 2
    return min(max_w, iter / (1000 / max_w))