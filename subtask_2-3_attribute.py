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

import pickle


def attribute_calculate(num_training_data, no_attr, model_single, device):
    my_dataloader = CelebALoader('/home/ubuntu/DL-lab/lab7/task_2/cglow')
    trainloader = DataLoader(my_dataloader, batch_size=1, shuffle=False)  
    dataset = iter(trainloader)

    c_pos = 0
    c_neg = 0
    
    with torch.no_grad():
        for i in range(num_training_data):
            print(i)
            image, cond = next(dataset)

            if cond[0][no_attr] == 1: 
    #         if cond[0][31] == 1 and cond[0][8] == 1: # smile and black_hair
                image = image.to(device)
                image = image_manual(image)
                _, _, z_data = model_single(image)

                c_pos += 1

                if c_pos == 1:             
                    z_avg_pos = z_data
                if c_pos != 1:
                    z_stack = []
                    z_avg_pos_old = z_avg_pos
                    z_avg_pos = []
                    for j in range(len(z_data)):
                        z_avg_pos.append((z_avg_pos_old[j]*(c_pos-1) + z_data[j])/c_pos)

            if cond[0][no_attr] != 1:
    #         if cond[0][31] != 1 and cond[0][8] != 1: # no smile and black_hair
                image = image.to(device)
                image = image_manual(image)            
                _, _, z_data = model_single(image)

                c_neg += 1
                if c_neg == 1:                 
                    z_avg_neg = z_data
                if c_neg != 1:
                    z_stack = []
                    z_avg_neg_old = z_avg_neg
                    z_avg_neg = []
                    for j in range(len(z_data)):                    
                        z_avg_neg.append((z_avg_neg_old[j]*(c_neg-1) + z_data[j])/c_neg)
    return z_avg_pos, z_avg_neg

def specify_imaages_z(path1, model_single, device):
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with open(os.path.join(path1), 'rb') as f:
        img1 = Image.open(f).convert('RGB')
    
    img1 = transform(img1).reshape((1,3,64,64))

    image = img1.to(device)
    image = image_manual(image)
    _, _, z_1 = model_single(image)


    return z_1


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
    parser.add_argument("--no_attr", type=int, default=31, help="use the no. of attribute") # smile = 1; Blond_Hair = 9; black_hair = 8
    parser.add_argument("--path1", type=str, default = "IMG_5544.jpg", help="specify the first image")
    parser.add_argument("--no_1", type=int, default=4, help="the first image from training data")
    parser.add_argument("--no_cuda", action="store_true")
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
    my_dataloader = CelebALoader('/home/ubuntu/DL-lab/lab7/task_2/cglow')
    trainloader = DataLoader(my_dataloader, batch_size=1, shuffle=False)  

    
    
    filename_pos = "z_avg_pos_" + str(args.no_attr) + ".p"
    filename_neg = "z_avg_neg_" + str(args.no_attr) + ".p"

    if os.path.isfile(filename_pos) & os.path.isfile(filename_neg):

        print(filename_pos + ' and ' + filename_neg + '資料夾已存在')
        z_avg_pos = pickle.load(open(filename_pos, "rb"))
        z_avg_neg = pickle.load(open(filename_neg, "rb"))
    else:   
        z_avg_pos, z_avg_neg = attribute_calculate(num_training_data = len(trainloader),  # len(trainloader)
                                                   no_attr = args.no_attr, 
                                                   model_single = model_single,
                                                   device = args.device)
        pickle.dump(z_avg_pos, file = open(filename_pos,"wb")) 
        pickle.dump(z_avg_neg, file = open(filename_neg,"wb"))
    
    
    if args.specify:
        z_data = specify_imaages_z(args.path1, model_single = model_single, device = args.device)
        
    if args.from_training_data:

        dataset = iter(trainloader)

        for ii in range(args.no_1): # no.1 + 1
            img1, _ = next(dataset)

        image = img1.to(args.device)
        image = image_manual(image)
        _, _, z_data = model_single(image)
             
    

    n = 10 # 內差 n 個 Z
    z_all = []
    z_all.append(z_data)
    for j in range(n):
        z_change = []
        for i in range(len(z_avg_pos)): # 4
            z_change.append(z_data[i] + torch.lerp(z_avg_pos[i], z_avg_neg[i], 0.1*(j+1)) - z_avg_pos[i])
        z_all.append(z_change)


    # model reverse 10 個 Z
    img_sample_all = 0
    with torch.no_grad():
        j == 0
        img_sample_all = model_single.reverse(z_all[0], reconstruct = True).cpu().data
        for j in range(n):
            a = model_single.reverse(z_all[j+1], reconstruct = True).cpu().data
            img_sample_all = torch.cat([img_sample_all, a], 0)
    save_image(img_sample_all, "result-3-attribute.png", nrow = n+2, normalize=True)

    
    
if __name__ == "__main__":
    main()
    