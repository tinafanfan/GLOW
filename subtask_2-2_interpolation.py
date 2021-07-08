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


def specify_imaages_z(path1, path2, model_single, device):
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with open(os.path.join(path1), 'rb') as f:
        img1 = Image.open(f).convert('RGB')

    with open(os.path.join(path2), 'rb') as f:
        img2 = Image.open(f).convert('RGB')
    
    img1 = transform(img1).reshape((1,3,64,64))
    img2 = transform(img2).reshape((1,3,64,64))
    img_sample_all = torch.cat([img1, img2], 0)

    image = img1.to(device)
    image = image_manual(image)
    _, _, z_1 = model_single(image)


    image = img2.to(device)
    image = image_manual(image)
    _, _, z_2 = model_single(image)

    return z_1, z_2

def image_manual(image):
    image = image * 255
    n_bits = 5
    n_bins = 2.0 ** n_bits
    if n_bits <  8:
        image = torch.floor(image/2 ** (8-n_bits))
    image = image / n_bins - 0.5
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_interpolation", type=int, default=10, help="number of interpolation images")
    parser.add_argument("--path1", type=str, default =  "IMG_5533.jpg", help="specify the first image")
    parser.add_argument("--path2", type=str, default =  "IMG_5534.jpg", help="specify the second image")
    parser.add_argument("--no_1", type=int, default=1, help="the first image from training data")
    parser.add_argument("--no_2", type=int, default=2, help="the second image from training data")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--random", action="store_true", help="random generate source images from random z")
    parser.add_argument("--specify", action="store_true", help="specify source images to conduct interpolation")
    parser.add_argument("--from_training_data", action="store_true", help="use training data as source images")
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model_single = Glow(3, 32, 4, affine=False, conv_lu=True)
    model = model_single.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load('checkpoint/optim_055001.pt'))
    model.load_state_dict(torch.load('checkpoint/model_055001.pt'))

    if args.random:
        z_shapes = calc_z_shapes(3, 64, 32, 4)

        # 產生兩個人
        z_sample_1 = []; z_sample_2 = []
        for i in z_shapes:
            z_new = torch.randn(1, *i) * 0.7
            z_sample_1.append(z_new.to(args.device))
        for i in z_shapes:
            z_new = torch.randn(1, *i) * 0.7
            z_sample_2.append(z_new.to(args.device))
       

    if args.specify:
        z_sample_1, z_sample_2 = specify_imaages_z(args.path1, args.path2, model_single = model_single, device = args.device)
    
    if args.from_training_data:
        my_dataloader = CelebALoader('/home/ubuntu/DL-lab/lab7/task_2/cglow')
        trainloader = DataLoader(my_dataloader, batch_size=1, shuffle=False)  
        dataset = iter(trainloader)

        for ii in range(args.no_1): # no.1 + 1
            img1, _ = next(dataset)

        image = img1.to(device)
        image = image_manual(image)
        _, _, z_sample_1 = model_single(image)
        
        for jj in range(args.no_2): # no.1 + 1
            img2, _ = next(dataset)

        image = img2.to(device)
        image = image_manual(image)
        _, _, z_sample_2 = model_single(image)        

    
    n = args.num_interpolation
    # 內差 n 個 Z
    z_sample_all = []
    z_sample_all.append(z_sample_1)
    for j in range(n):
        z_sample_int = []
        for i in range(len(z_sample_1)): # 4
            z_sample_int.append(torch.lerp(z_sample_1[i], z_sample_2[i], 0.1*(j+1)))
        z_sample_all.append(z_sample_int)
    z_sample_all.append(z_sample_2)


    # model reverse 10 個 Z
    img_sample_all = 0
    with torch.no_grad():
        j == 0
        img_sample_all = model_single.reverse(z_sample_all[0], reconstruct = True).cpu().data
        for j in range(n+1):
            a = model_single.reverse(z_sample_all[j+1], reconstruct = True).cpu().data
            img_sample_all = torch.cat([img_sample_all, a], 0)  # img_sample_all.size() -> img_sample_all
    save_image(img_sample_all, "result-2-interpolation.png", nrow = n+2, normalize=True)

if __name__ == "__main__":
    main()









