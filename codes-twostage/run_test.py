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

def testing(test_dataset, models, device, args):
    model1, model2 = models
    model1.eval()
    model2.eval()
    threshold = args.conf_thresh
    total_cls = torch.zeros(5).to(device)
    tp_cls = torch.zeros(5).to(device)
    fp_cls = torch.zeros(5).to(device)

    #recordcsv = open(args.recordtxtname,'w')
    #writer = csv.writer(recordcsv)

    post_pos = torch.ones((5)).to(device)
    post_neg = torch.zeros((5)).to(device)
    with torch.no_grad():
        for idx in range(int(0.2*len(test_dataset.dirname_list))):
            images, label, paths = test_dataset[idx]
            images, label = images.to(device), label.to(device)

            ''' up to 75.26'''
            output1 = nn.Sigmoid()(model1(images))
            output1 = torch.where(output1>threshold, 1., 0.)
            post_pos = torch.where(torch.sum(output1, dim=0)>0, 1., 0.)

            output = nn.Sigmoid()(model2(images))*post_pos
            #output = torch.where(output1==1., (output+output1)/2., output)      # up to 76.46
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

            # for record.txt
            '''
            output1 = nn.Sigmoid()(model1(images))
            output1 = torch.where(output1>threshold, 1., 0.)
            tmp1 = output1.clone().cpu().numpy().astype(int)
            post_pos = torch.where(torch.sum(output1, dim=0)>0, 1., 0.)
            output = nn.Sigmoid()(model2(images))*post_pos
            tmp2 = torch.where(output>threshold, 1., 0.).clone().cpu().numpy().astype(int)
            output = torch.where(output1==1., (output+output1)/2., output)      # up to 76.46
            output = torch.where(output>threshold, 1., 0.)
            '''

            ''' up to 75.92'''
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

            # for record.txt
            '''
            tmp3 = output.clone().cpu().numpy().astype(int)
            tmp0 = label.clone().cpu().numpy().astype(int)
            writer.writerow(['dir'])
            tmp = [['label']+list(i)+['final']+list(j)+['prec']+list(k)+['rec']+list(l) for i, j, k, l in zip(tmp0, tmp3, tmp1, tmp2)]
            writer.writerows(tmp)
            '''

            tp_batch = label*output
            fp_batch = output-tp_batch 
            total_cls = total_cls + torch.sum(label,dim=0)
            tp_cls = tp_cls + torch.sum(tp_batch, dim=0)
            fp_cls = fp_cls + torch.sum(fp_batch, dim=0)

            for i, path in enumerate(paths):
                ID = os.path.split(path)[-1]
                dirname = ID.split('_')[0]
                pred = list(output[i].cpu().numpy().astype(int))
                csvlist.append([dirname, ID]+pred)

    #recordcsv.close()
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
    print('')
    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(csvlist)
    print('test csv predicted file has been written as '+csv_path)

def testing_seq(test_dataset, model, device, args):
    model.eval()
    threshold = args.conf_thresh
    total_cls = torch.zeros(5).to(device)
    tp_cls = torch.zeros(5).to(device)
    fp_cls = torch.zeros(5).to(device)

    #recordcsv = open(args.recordtxtname,'w')
    #writer = csv.writer(recordcsv)

    post_pos = torch.ones((5)).to(device)
    post_neg = torch.zeros((5)).to(device)
    with torch.no_grad():
        for idx in range(int(0.2*len(test_dataset.dirname_list))):
            images, label, paths = test_dataset[idx]
            images, label = images.to(device), label.to(device)

            output = nn.Sigmoid()(model(images))
            output = torch.where(output>threshold, 1., 0.)

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

            tp_batch = label*output
            fp_batch = output-tp_batch 
            total_cls = total_cls + torch.sum(label,dim=0)
            tp_cls = tp_cls + torch.sum(tp_batch, dim=0)
            fp_cls = fp_cls + torch.sum(fp_batch, dim=0)

    #recordcsv.close()
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
    print('')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="data_dir is the data path used here", default='../Blood_data/train')
    parser.add_argument("--csv-path", help="csv-path is the data label csv path used here", default='../Blood_data/train.csv')
    parser.add_argument("--modelpre-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--modelrec-path", help="used model name to compute", default='./resnet.pth')
    parser.add_argument("--conf-thresh", help="threshold of the predicted confidence", type=float, default=0.8)
    parser.add_argument("--recordtxtname", default='./record.txt')
    args = parser.parse_args()

    dataset = medical_dataset(args.data_dir, csv_path=args.csv_path, type='validation')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_recall = models.resnet18().to(DEVICE)
    model_recall, _ = load_model(args.modelrec_path, model_recall, device=DEVICE)

    model_precision = models.resnet18().to(DEVICE)
    model_precision, _ = load_model(args.modelpre_path, model_precision, device=DEVICE)

    testing(
            test_dataset=dataset,
            models=(model_precision, model_recall),
            device=DEVICE,
            args=args,
            )
    '''

    testing_seq(
            test_dataset=dataset,
            model=model_recall,
            device=DEVICE,
            args=args,
            )

    '''
