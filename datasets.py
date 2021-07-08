import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision.transforms as transforms



def preprocess(x, scale, bias, bins, noise=False):

    x = x / scale
    x = x - bias

    if noise == True:
        if bins == 2:
            x = x + torch.zeros_like(x).uniform_(-0.5, 0.5)
        else:
            x = x + torch.zeros_like(x).uniform_(0, 1/bins)
    return x


def postprocess(x, scale, bias):

    x = x + bias
    x = x * scale
    return x


def convert_to_img(y):
    import skimage.color
    import skimage.util
    import skimage.io

    C = y.size(1)

    transform = transforms.ToTensor()
    colors = np.array([[0,0,0],[255,255,255]])/255

    if C == 1:
        seg = torch.squeeze(y, dim=1).cpu().numpy()
        seg = np.nan_to_num(seg)
        seg = np.clip(np.round(seg),a_min=0, a_max=1)

    if C > 1:
        seg = torch.mean(y, dim=1, keepdim=False).cpu().numpy()
        seg = np.nan_to_num(seg)
        seg = np.clip(np.round(seg),a_min=0, a_max=1)

    B,C,H,W = y.size()
    imgs = list()
    for i in range(B):
        label_i = skimage.color.label2rgb(seg[i], colors=colors, bg_label = 0)
        label_i = skimage.util.img_as_ubyte(label_i)
        imgs.append(transform(label_i))
    return imgs, seg


class HorseDataset(data.Dataset):

    def __init__(self, dir, size, n_c, portion="train"):
        self.dir = dir
        self.names = self.read_names(dir, portion)
        self.n_c = n_c
        self.size = size

    def read_names(self, dir, portion):

        path = os.path.join(dir, "{}.txt".format(portion))
        names = list()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                name = {}
                name["img"] = os.path.join(dir, os.path.join("images", line))
                name["lbl"] = os.path.join(dir, os.path.join("labels", line))
                names.append(name)
        return names

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):

        # path
        name = self.names[index]
        img_path = name["img"]
        lbl_path = name["lbl"]
        transform = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])

        # img
        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        # lbl
        lbl = Image.open(lbl_path).convert("L")
        lbl = transform(lbl)
        lbl = torch.round(lbl)
        if self.n_c > 1:
            lbl = lbl.repeat(self.n_c,1,1)

        return {"x":img, "y":lbl}
    
#################################################################################
def get_CelebA_data(root_folder):
    img_list = sorted(os.listdir(os.path.join(root_folder, 'CelebA-HQ-img')), key=lambda x: int(x[:-4]))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False):
        self.root_folder = root_folder
        # assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        
        self.cond = cond
        self.img_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)


    def __getitem__(self, index):
        path = os.path.join(self.root_folder, "CelebA-HQ-img", self.img_list[index])
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        label = self.label_list[index]
        
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        img = transform(img)
        label = torch.Tensor(label)
        
        return img, label

def get_CelebA_data_prediction(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img_prediction'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno_prediction.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader_prediction(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False):
        self.root_folder = root_folder
        # assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        
        self.cond = cond
        self.img_list, self.label_list = get_CelebA_data_prediction(self.root_folder)
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)


    def __getitem__(self, index):
        path = os.path.join(self.root_folder, "CelebA-HQ-img_prediction", self.img_list[index])
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        label = self.label_list[index]
        
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        img = transform(img)
        label = torch.Tensor(label)
        
        return img, label
#################################################################################



