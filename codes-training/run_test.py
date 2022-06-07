import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

import models
from data_prepare import medical_dataset, GeneratorSampler
from utils import save_model, load_model

def testing(val_loader, model, device, args):
    model.eval()
    batch_size = args.batch_size
    threshold = args.conf_thresh
    total_cls = torch.zeros(5).to(device)
    tp_cls = torch.zeros(5).to(device)
    fp_cls = torch.zeros(5).to(device)
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            images, label = data
            images, label = images.to(device), label.to(device)
            output = nn.Sigmoid()(model(images))
            output = torch.where(output>threshold, 1., 0.)

            tp_batch = label*output
            fp_batch = output-tp_batch 
            total_cls = total_cls + torch.sum(label,dim=0)
            tp_cls = tp_cls + torch.sum(tp_batch, dim=0)
            fp_cls = fp_cls + torch.sum(fp_batch, dim=0)


    tp_cls, fp_cls, total_cls = tp_cls.cpu().numpy(), fp_cls.cpu().numpy(), total_cls.cpu().numpy()
    tp = int(np.sum(tp_cls))
    fp = int(np.sum(fp_cls))
    total = int(np.sum(total_cls))
    if tp+fp==0: pre=0
    else: pre = tp/(tp+fp)
    rec = tp/total
    if pre==0 and rec==0: f2=0
    else: f2 = (1+2**2)*(rec*pre)/(rec+2**2*pre)
    acc_class = tp_cls/total_cls*100

    print('\n[val set]: Precision = {:.2f}%\tRecall = {:.2f}%\tf2-score = {:.2f}%'.format(pre*100, rec*100, f2*100))
    print('Accuracy per class:')
    print('[ich] - {:.2f}%'.format(acc_class[0]))
    print('[ivh] - {:.2f}%'.format(acc_class[1]))
    print('[sah] - {:.2f}%'.format(acc_class[2]))
    print('[sdh] - {:.2f}%'.format(acc_class[3]))
    print('[edh] - {:.2f}%'.format(acc_class[4]))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="data_dir is the data path used here", default='../Blood_data/train')
    parser.add_argument("--csv-path", help="csv-path is the data label csv path used here", default='../Blood_data/train.csv')
    parser.add_argument("--batch-size", help="batch size for every iteration", type=int, default=64)
    parser.add_argument("--model-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--conf-thresh", help="threshold of the predicted confidence", type=float, default=0.8)
    args = parser.parse_args()

    dataset = medical_dataset(args.data_dir, csv_path=args.csv_path, type='validation')
    val_loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        num_workers=1,
                        sampler=GeneratorSampler(args.csv_path, type='validation'),
                        )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18().to(DEVICE)
    model, _ = load_model(args.model_path, model, device=DEVICE)

    #if torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)

    testing(
            val_loader=val_loader,
            model=model,
            device=DEVICE,
            args=args,
            )

