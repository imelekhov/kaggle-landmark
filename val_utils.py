import gc
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import os
from sklearn.metrics import accuracy_score

def validate_epoch(net, val_loader, criterion):
    net.eval()

    running_loss = 0.0
    n_batches = len(val_loader)
    sm = nn.Softmax(1)
    acc = np.zeros(n_batches)
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    
    for i, sample in pbar:
        labels = Variable(sample['label'].long().cuda(), volatile=True)
        inputs = Variable(sample['img'].cuda(), volatile=True)
        
        outputs = net(inputs)
        
        if inputs.size(0) != torch.cuda.device_count():
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        targets = sample['label'].numpy()
        preds = sm(outputs).data.cpu().numpy().argmax(1)
        acc[i] = accuracy_score(targets, preds)

        running_loss += loss.data.cpu().numpy()[0]
        pbar.set_description('Running_loss: %.3f / total_loss: %.3f' % (running_loss / (i+1), loss.data.cpu().numpy()[0]))
        gc.collect()
    gc.collect()

    return running_loss/n_batches, acc.mean()
