import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import pandas as pd
from PIL import Image

import os
import glob
import random
import time
import numpy as np

class RandomGaussian(nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 1.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    def forward(self, img):
        if torch.rand(1) < self.p:
            return transforms.GaussianBlur(self.kernel_size, self.sigma)(img)
        return img
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class medical_dataset(Dataset):
    def __init__(self, data_dir, csv_path=None, type='train'):
        self.images = None
        self.data_dir = data_dir
        self.csv_path = csv_path

        if not csv_path:
            self.dirname_list = [i for i in os.listdir(data_dir) if i[:2]=='ID']
        else:
            self.data_df = pd.read_csv(csv_path)
            self.dir_dict = dict()
            for index in range(len(self.data_df)):
                dirname = self.data_df.loc[index, "dirname"]
                ID = self.data_df.loc[index, "ID"]
                if dirname in self.dir_dict.keys():
                    self.dir_dict[dirname].append(index)
                else:
                    self.dir_dict[dirname] = [index]

            self.dirname_list = list(self.dir_dict.keys())
            random.seed(1)
            random.shuffle(self.dirname_list)

        self.transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((512,512)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.1592],[0.0651]),
                    ])


    def __getitem__(self, index):
        dirname = self.dirname_list[index]
        if not self.csv_path:
            filenames = glob.glob(os.path.join(self.data_dir, dirname, '*.jpg'))
            filenames = sorted(filenames, key=lambda y: int(y.split('_')[-1][:-4]))
            for path in filenames:
                try:
                    image = torch.cat((image, self.transform(path).unsqueeze(0)), dim=0)
                    paths.append(path)
                except:
                    image = self.transform(path).unsqueeze(0)
                    paths = [path]
            return image, paths
        else:
            dirlabel = torch.torch.zeros((1,5))
            for i, dirID in enumerate(self.dir_dict[dirname]):
                path = os.path.join(self.data_dir, self.data_df.loc[dirID, "dirname"], self.data_df.loc[dirID, "ID"])
                label = torch.zeros((1,5))
                label[0,0] = float(self.data_df.loc[dirID, "ich"])
                label[0,1] = float(self.data_df.loc[dirID, "ivh"])
                label[0,2] = float(self.data_df.loc[dirID, "sah"])
                label[0,3] = float(self.data_df.loc[dirID, "sdh"])
                label[0,4] = float(self.data_df.loc[dirID, "edh"])
                if i>0:
                    dirlabel = torch.cat((dirlabel,label), dim=0)
                    image = torch.cat((image, self.transform(path).unsqueeze(0)), dim=0)
                    paths.append(path)
                else:
                    dirlabel = label
                    image = self.transform(path).unsqueeze(0)
                    paths = [path]

            return image, dirlabel, paths

    def __len__(self):
        return len(self.data_df) if self.csv_path else None

if __name__=='__main__':
    data_dir = '../Blood_data/train'
    csv_path = '../Blood_data/train.csv'

    '''
    datasetval = medical_dataset(data_dir, csv_path=csv_path, type='validation')
    validate_len = int(0.2*len(datasetval.dirname_list))
    num = 0
    for i in range(validate_len):
        dirname, image, label = datasetval[i]
        print(dirname)
        print(label)
        num = num+image.shape[0]
        if num>=50:break
    print('total image:', num)
    print('total dir  :',i+1)

        
    for i, data in enumerate(valiter):
        image, label = data
        print('what is image:', image.squeeze(0).shape)
        print('what is label:', label.squeeze(0).shape)
        print('')
        if i==5:break
    '''

    datasettest = medical_dataset('../Blood_data/test', type='pred')
    for idx, data in enumerate(datasettest):
        images, paths = data
        print(images.shape)
        for i in paths:
            print(i)
        if idx==0:break
