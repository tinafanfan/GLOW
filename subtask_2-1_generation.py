from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from datasets import CelebALoader

from model import Glow
from torchvision.utils import save_image
import gc

import torchvision
import matplotlib.pyplot as plt

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_generation", type=int, default=10, help="number of interpolation images")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device


    model_single = Glow(3, 32, 4, affine=False, conv_lu=True)
    # model = nn.DataParallel(model_single)
    model = model_single.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load('checkpoint/optim_055001.pt'))
    model.load_state_dict(torch.load('checkpoint/model_055001.pt'))

    z_shapes = calc_z_shapes(3, 64, 32, 4)

    m = args.num_generation # 產生 m 個人

    z_sample_all = []
    for j in range(m):
        z_sample_1 = []
        for i in z_shapes:
            a = torch.randn(1, *i)
            z_new = torch.randn(1, *i) * 0.7
            z_sample_1.append(z_new.to(args.device))
        z_sample_all.append(z_sample_1)

    # model reverse 10 個 Z
    img_sample_all = 0
    with torch.no_grad():
        j == 0
        img_sample_all = model_single.reverse(z_sample_all[0]).cpu().data
        for j in range(m - 1):
            a = model_single.reverse(z_sample_all[j+1]).cpu().data
            img_sample_all = torch.cat([img_sample_all, a], 0)
    save_image(img_sample_all, "result-1-generation.png", nrow = m, normalize=True)
    
    
if __name__ == "__main__":
    main()


    