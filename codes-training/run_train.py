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

def training(loaders, model_param, device, args):
    train_loader, val_loader = loaders
    batch_size, epoch, st_epoch, log_interval, save_path = args.batch_size, args.epoch, args.start_ep, args.log_interval, args.save_path
    LR = args.lr
    lam = args.lam   #positive loss weight

    criterion = nn.BCELoss()
    #pos_weight = torch.Tensor([7.65, 11.51, 7.94, 5.07, 22.55]).to(device)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model, optimizer = model_param
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))

    model.train()

    iteration=1
    for ep in range(st_epoch, epoch):
        for idx, data in enumerate(train_loader):
            images, label = data
            images, label = images.to(device), label.to(device).float()
            output = nn.Sigmoid()(model(images)).float()

            pos_index = torch.nonzero((label==1.), as_tuple=True)
            neg_index = torch.nonzero((label==0.), as_tuple=True)
            pos_label, pos_output, neg_label, neg_output = label[pos_index], output[pos_index], label[neg_index], output[neg_index]

            pos_loss = criterion(pos_output, pos_label)
            neg_loss = criterion(neg_output, neg_label)
            loss = lam*pos_loss + (1-lam)*neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if iteration%log_interval==0:
                print('ep: {:d} [{:d}/{:d}]\tloss: {:4f}'.format(ep, (idx+1)*batch_size, len(train_loader.sampler), loss.item()))

            iteration+=1

        model.eval()
        testing(val_loader, model, device, args)
        model.train()
        save_model(os.path.join(save_path,'resnet-ep{:d}.pth'.format(ep)), model, optimizer)

def testing(val_loader, model, device, args):
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
    parser.add_argument("--start-ep", help='epoch has been run', type=int, default=0)
    parser.add_argument("--epoch", help="epoch is num of episode in a training", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=5e-4)
    parser.add_argument("--lam", help="positive loss weight", type=float, default=0.8)
    parser.add_argument("--log-interval", help="show training log every x iterations", type=int, default=100)
    parser.add_argument("--conf-thresh", help="threshold of the predicted confidence", type=float, default=0.8)
    parser.add_argument("--save-path", help="threshold of the predicted confidence", default='./backup/')
    args = parser.parse_args()

    dataset = medical_dataset(args.data_dir, csv_path=args.csv_path, type='train')
    train_loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        num_workers=1,
                        sampler=GeneratorSampler(args.csv_path, type='train'),
                        )
    datasetval = medical_dataset(args.data_dir, csv_path=args.csv_path, type='validation')
    val_loader = DataLoader(
                        datasetval,
                        batch_size=args.batch_size,
                        num_workers=1,
                        sampler=GeneratorSampler(args.csv_path, type='validation'),
                        )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18().to(DEVICE)
    if args.start_ep > 0:
        model, optimizer = load_model('./backup/resnet-ep'+str(args.start_ep-1)+'.pth', model, device=DEVICE)
    else:
        optimizer = None
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    training(
            loaders=(train_loader, val_loader),
            model_param=(model, optimizer),
            device=DEVICE,
            args=args,
            )

    '''
    w_ich = 1/5911
    w_ivh = 1/4132
    w_sah = 1/5843
    w_sdh = 1/8307
    w_edh = 1/2017
    re_w = lam/(w_ich+w_ivh+w_sah+w_sdh+w_edh)
    w_ich = re_w*w_ich
    w_ivh = re_w*w_ivh
    w_sah = re_w*w_sah
    w_sdh = re_w*w_sdh
    w_edh = re_w*w_edh
    print(w_ich+w_ivh+w_sah+w_sdh+w_edh)
            ich_index = torch.nonzero((label[:,0]==1.), as_tuple=True)
            ich_index = (ich_index[0], torch.zeros(len(ich_index[0]), dtype=int))
            ivh_index = torch.nonzero((label[:,1]==1.), as_tuple=True)
            ivh_index = (ivh_index[0], torch.ones(len(ivh_index[0]), dtype=int))
            sah_index = torch.nonzero((label[:,2]==1.), as_tuple=True)
            sah_index = (sah_index[0], 2*torch.ones(len(sah_index[0]), dtype=int))
            sdh_index = torch.nonzero((label[:,3]==1.), as_tuple=True)
            sdh_index = (sdh_index[0], 3*torch.ones(len(sdh_index[0]), dtype=int))
            edh_index = torch.nonzero((label[:,4]==1.), as_tuple=True)
            edh_index = (edh_index[0], 4*torch.ones(len(edh_index[0]), dtype=int))
            neg_index = torch.nonzero((label==0.), as_tuple=True)

            ich_loss = criterion(output[ich_index], label[ich_index])
            ivh_loss = criterion(output[ivh_index], label[ivh_index])
            sah_loss = criterion(output[sah_index], label[sah_index])
            sdh_loss = criterion(output[sdh_index], label[sdh_index])
            edh_loss = criterion(output[edh_index], label[edh_index])
            neg_loss = criterion(output[neg_index], label[neg_index])
            loss = w_ich*ich_loss + w_ivh*ivh_loss + w_sah*sah_loss + w_sdh*sdh_loss + w_edh*edh_loss + (1-lam)*neg_loss
    '''
