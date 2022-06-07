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
from data_prepare import medical_dataset, GeneratorSampler
from utils import save_model, load_model

def predicting(test_loader, model, device, args):
    model.eval()
    batch_size, threshold, csv_path = args.batch_size, args.conf_thresh, args.out_csv

    csvlist = list()
    csvlist.append(['dirname', 'ID', 'ich', 'ivh', 'sah', 'sdh', 'edh'])
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, paths = data
            images = images.to(device)
            output = nn.Sigmoid()(model(images))
            output = torch.where(output>threshold, 1., 0.)

            for i, path in enumerate(paths):
                ID = os.path.split(path)[-1]
                dirname = ID.split('_')[0]
                pred = list(output[i].cpu().numpy().astype(int))
                csvlist.append([dirname, ID]+pred)

    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(csvlist)
    print('test csv predicted file has been written')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="data_dir is the data path used here", default='../Blood_data/train')
    parser.add_argument("--batch-size", help="batch size for every iteration", type=int, default=64)
    parser.add_argument("--model-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--conf-thresh", help="threshold of the predicted confidence", type=float, default=0.8)
    parser.add_argument("--out-csv", help="output csv path", default='../test_pred.csv')
    args = parser.parse_args()

    dataset = medical_dataset(args.data_dir, type='pred')
    test_loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        num_workers=1,
                        )

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = models.resnet18().to(DEVICE)
    model, _ = load_model(args.model_path, model, device=DEVICE)

    predicting(
            test_loader=test_loader,
            model=model,
            device=DEVICE,
            args=args,
            )

