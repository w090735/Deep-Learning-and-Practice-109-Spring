import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from math import log, pi, exp,sqrt
from scipy import linalg as la

from tqdm import tqdm

import argparse

from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

def set_args():
	'''
		icelvr
	'''
	parser = argparse.ArgumentParser(description="cNF")
	parser.add_argument("--batch", default=16, type=int, help="batch size")
	parser.add_argument("--iter", default=1000, type=int, help="maximum iterations") #200000
	parser.add_argument(
	    "--n_flow", default=32, type=int, help="number of flows in each block"
	)
	parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
	parser.add_argument(
	    "--no_lu",
	    action="store_true",
	    help="use plain convolution instead of LU decomposed version",
	)
	parser.add_argument(
	    "--affine", action="store_true", help="use affine coupling instead of additive"
	)
	parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
	parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
	parser.add_argument("--img_size", default=32, type=int, help="image size") # 64
	parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
	parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
	parser.add_argument("--path", default="/home/auser03/lab7/task_2/", type=str, jelp="path of dataset")
	args = parser.parse_args(args=[])

	return args

def set_test_args():
	parser = argparse.ArgumentParser(description="test args of cNF")
    parser.add_argument('-f', "--filelist_path")
    parser.add_argument('-p', '--ckpt_path', default="checkpoint/model_190000.pt")
    parser.add_argument('-c', '--cond_data', default="inference_data/for_inference.pt")
    parser.add_argument('-o', "--output_dir", default="sample/")
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument('-n', "--num_samples", default=8)
    parser.add_argument('-t', "--feature", default="Smiling")
    parser.add_argument('-i', "--index", default=1)

    args = parser.parse_args()

    return args

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    # compute z shape in each block
    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    # compute z shape in last block
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    # total pixel in one image
    n_pixel = image_size * image_size * 3

    # avoid gradient vanish
    loss = -log(n_bins) * n_pixel
    # compute total loss
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )
