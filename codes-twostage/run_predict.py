import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import argparse

import models
from data_prepare import medical_dataset
from utils import save_model, load_model

def predicting(test_dataset, models, device, args):
    model1, model2 = models
    model1.eval()
    model2.eval()
    threshold, csv_path = args.conf_thresh, args.out_csv

    csvlist = list()
    csvlist.append(['dirname', 'ID', 'ich', 'ivh', 'sah', 'sdh', 'edh'])
    post_pos = torch.ones((5)).to(device)
    post_neg = torch.zeros((5)).to(device)
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            images, paths = data
            images = images.to(device)

            output1 = nn.Sigmoid()(model1(images))
            output1 = torch.where(output1>threshold, 1., 0.)
            post_pos = torch.where(torch.sum(output1, dim=0)>0, 1., 0.)
            
            output = nn.Sigmoid()(model2(images))*post_pos
            #output = torch.where(output1==1., (output+output1)/2., output)
            #output = torch.where(output>threshold, 1., 0.)

            for i, v in enumerate(post_pos):
                if v==1.:
                    con_1, con_2 = False, False
                    for j in range(int(output.shape[0]/2)):
                        if con_1 and con_2:
                            break
                        if output1[j, i]==1.:
                            con_1 = True
                        elif (output1[j, i]==0.) and (not con_1):
                            output[j, i] = (output[j, i]+0.65)/2
                        if output1[output.shape[0]-1-j, i]==1.:
                            con_2 = True
                        elif (output1[output.shape[0]-1-j, i]==0.) and (not con_2):
                            output[output.shape[0]-1-j, i] = (output[output.shape[0]-1-j, i]+0.65)/2
            output = torch.where(output1==1., (output+output1)/2., output)      # up to 76.46
            output = torch.where(output>0.75, 1., 0.)

            output[0] = torch.where((output[1]+output[2])==0., post_neg, output[0])
            for i in range(3, output.shape[0]-4):
                if i < output.shape[0]-2:
                    output[i] = torch.where((output[i-1]+output[i+2])==2., post_pos, output[i])
                output[i] = torch.where((output[i-1]+output[i+1])==2., post_pos, output[i])
                if i < output.shape[0]-3:
                    output[i] = torch.where((((output[i-1]+output[i+1])==0.) & ((output[i+2]+output[i+3])==0.)), post_neg, output[i])
                else:
                    output[i] = torch.where(((output[i-1]+output[i+1])==0.), post_neg, output[i])
            output[-1] = torch.where(output[-2]==0., post_neg, output[-1])

            for i, path in enumerate(paths):
                ID = os.path.split(path)[-1]
                dirname = ID.split('_')[0]
                pred = list(output[i].cpu().numpy().astype(int))
                csvlist.append([dirname, ID]+pred)

    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(csvlist)
    print('test csv predicted file has been written as '+csv_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="data_dir is the data path used here", default='../Blood_data/train')
    parser.add_argument("--modelpre-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--modelrec-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--conf-thresh", help="threshold of the predicted confidence", type=float, default=0.8)
    parser.add_argument("--out-csv", help="output csv path", default='../test_pred.csv')
    args = parser.parse_args()

    dataset = medical_dataset(args.data_dir, type='pred')

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_recall = models.resnet18().to(DEVICE)
    model_recall, _ = load_model(args.modelrec_path, model_recall, device=DEVICE)

    model_precision = models.resnet18().to(DEVICE)
    model_precision, _ = load_model(args.modelpre_path, model_precision, device=DEVICE)

    predicting(
            test_dataset=dataset,
            models=(model_precision, model_recall),
            device=DEVICE,
            args=args,
            )

