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
            self.filenames = []
            dirnames = [i for i in os.listdir(data_dir) if i[:2]=='ID']
            for dirname in dirnames:
                filenames = glob.glob(os.path.join(data_dir, dirname, '*.jpg'))
                for fn in filenames:
                    self.filenames.append((fn))
        else:
            self.data_df = pd.read_csv(csv_path)

        if type=='train':
            self.transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((512,512)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(15),
                    transforms.RandomVerticalFlip(p=0.5),
                    RandomGaussian(5),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.1592],[0.0651]),
                    ])
        else:
            self.transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((512,512)),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.1592],[0.0651]),
                    ])


    def __getitem__(self, index):
        if not self.csv_path:
            path = self.filenames[index]
            image = self.transform(path)
            return image, path
        else:
            dirname = self.data_df.loc[index, "dirname"]
            path = os.path.join(self.data_dir, self.data_df.loc[index, "dirname"], self.data_df.loc[index, "ID"])
            label = torch.zeros(5)
            label[0] = float(self.data_df.loc[index, "ich"])
            label[1] = float(self.data_df.loc[index, "ivh"])
            label[2] = float(self.data_df.loc[index, "sah"])
            label[3] = float(self.data_df.loc[index, "sdh"])
            label[4] = float(self.data_df.loc[index, "edh"])

            image = self.transform(path)
            return image, label

    def __len__(self):
        return len(self.data_df) if self.csv_path else len(self.filenames)

class GeneratorSampler(Sampler):
    def __init__(self, csv_path, type='train'):
        self.dir_dict = dict()
        self.sampled_point = list()
        #self.edh = list()
        self.data_df = pd.read_csv(csv_path)

        for index in range(len(self.data_df)):
            dirname = self.data_df.loc[index, "dirname"]
            ID = self.data_df.loc[index, "ID"]
            if dirname in self.dir_dict.keys():
                self.dir_dict[dirname].append(index)
            else:
                self.dir_dict[dirname] = [index]

        dirname_list = list(self.dir_dict.keys())
        random.seed(1)
        random.shuffle(dirname_list)
        validate_len = int(0.1*len(dirname_list))

        if type=='train':
            for dirname in dirname_list[validate_len:]:
                for index in self.dir_dict[dirname]:
                    self.sampled_point.append(index)
                    #if self.data_df.loc[index, "edh"] == 1.0:
                        #self.edh.append(index)
            #self.sampled_point = self.sampled_point+self.edh
            t = 1000*time.time()
            random.seed(int(t)%2**32)
            random.shuffle(self.sampled_point)

        elif type=='validation':
            for dirname in dirname_list[:validate_len]:
                for index in self.dir_dict[dirname]:
                    self.sampled_point.append(index)

    def __iter__(self):
        return iter(self.sampled_point)

    def __len__(self):
        return len(self.sampled_point)

if __name__=='__main__':
    data_dir = '../Blood_data/train'
    csv_path = '../Blood_data/train.csv'
    dataset = medical_dataset(data_dir, csv_path=csv_path, type='train')
    train_loader = DataLoader(
                        dataset,
                        batch_size=16,
                        num_workers=1,
                        sampler=GeneratorSampler(csv_path, type='train'),
                        )
    datasetval = medical_dataset(data_dir, csv_path=csv_path, type='validation')
    val_loader = DataLoader(
                        datasetval,
                        batch_size=32,
                        num_workers=1,
                        sampler=GeneratorSampler(csv_path, type='validation'),
                        )

    image, label = iter(train_loader).next()
    valimage, vallabel = iter(val_loader).next()
    print('a batch of train loader: (image/label)',image.shape, label.shape)
    print('a batch of validation loader: (image/label)',valimage.shape, vallabel.shape)
    print('total train files:', len(train_loader.sampler))
    print('total val files:', len(val_loader.sampler))
    print('')

    '''
    test_dataset = medical_dataset('../Blood_data/test', type='test')
    test_loader = DataLoader(
                        test_dataset,
                        batch_size=16,
                        shuffle=False,
                        num_workers=1,
                        )
    image, path = iter(test_loader).next()
    print('a batch of test loader image:',image.shape)
    '''

    ##############
    # compute mean and std
    ##############
    '''
    imagemean = None
    for image, label in train_loader:
        mean = torch.mean(image, dim=(2,3))# shape: (64, 3, 512, 512)
        std = torch.std(image, dim=(2,3))
        if imagemean is None:
            imagemean = mean
            imagestd = std
        else:
            imagemean = torch.cat((imagemean, mean), dim=0)
            imagestd = torch.cat((imagestd, std), dim=0)
    print(imagemean.shape)
    print(imagestd.shape)
    meanvalue = torch.mean(imagemean, dim=0)
    stdvalue = torch.std(imagestd, dim=0)
    print(meanvalue, stdvalue)
    '''
#Ich=5911
#Ivh=4132
#Sah=5843
#Sdh=8307   /2
#Edh=2017   *2
