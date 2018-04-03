from __future__ import print_function


from dataset import CameraClassificationTrainDataset, get_class_dict

from val_utils import validate_epoch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import argparse
import os, fnmatch
import pandas as pd
import sys
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F

import gc

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix

import time
import pickle

from collections import OrderedDict

import time
cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transfroms

import model_zoo
from augmentation import *

def find_checkpoint(path, pattern='*.pth'):
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return name
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='./data')
    parser.add_argument('--snapshots_path',  default='snapshots/')
    parser.add_argument('--snapshot',  default='')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--TTA', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=422345)
    args = parser.parse_args()

    tmp = np.load(os.path.join(args.snapshots_path, 'mean_std.npy'))
    mean_vector, std_vector = tmp
    normTransform = transforms.Normalize(torch.from_numpy(mean_vector), torch.from_numpy(std_vector))

    os.makedirs('val_predicts',exist_ok=True)

    with open(os.path.join(args.snapshots_path, args.snapshot, 'args.pkl'), 'rb') as f:
        tmp = pickle.load(f)
        args.experiment = tmp.experiment
        args.crop_size = tmp.crop_size
        args.drop = tmp.drop
        args.seed = tmp.seed
        
    if args.TTA:
        val_transforms = transforms.Compose([
          lambda x: center_crop(x, args.crop_size),
          lambda x: five_crop(x, int(args.crop_size*0.95)),
          lambda crops: torch.stack([normTransform(transforms.ToTensor()(crop)) for crop in crops])
        ])
    else:
        val_transforms = transforms.Compose([
          lambda x: center_crop(x, args.crop_size),
          transforms.ToTensor(),
          normTransform
        ])
    
    # Preparing the dataset
    data_frame = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
    camera_id_train = data_frame['camera'].values
    fname_train = data_frame['fname'].values

    # Preparing the matrices and shuffle them
    ind = np.arange(camera_id_train.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(ind)
    camera_id_train_shuffled = camera_id_train[ind]
    fname_train_shuffled = fname_train[ind]


    results = 0
    skf = StratifiedKFold(n_splits=args.n_folds, random_state=args.seed)
    preds = []
    gt = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(fname_train_shuffled, camera_id_train)):
        X, X_val = fname_train_shuffled[train_index], fname_train_shuffled[test_index]
        Y, Y_val = camera_id_train_shuffled[train_index], camera_id_train_shuffled[test_index]

        val_ds = CameraClassificationTrainDataset(args.dataset, fname_split=X_val, classes_split=Y_val, transform=val_transforms)

        val_loader = data.DataLoader(val_ds,
                                     batch_size=args.bs,
                                     num_workers=args.n_threads,
                                     sampler=data.sampler.SequentialSampler(val_ds)
                                    )

        print(colored('==> ', 'red')+'fold: ', fold_id)

        

        path = os.path.join(args.snapshots_path, args.snapshot, 'fold_'+str(fold_id))
        checkpoint_name = find_checkpoint(path)
        checkpoint = torch.load(os.path.join(path, checkpoint_name))

        net = model_zoo.get_model(args)
        net.load_state_dict(checkpoint)
        net = nn.DataParallel(net)
        net.cuda()
        net.eval()
        
        # Getting the fold predictions
        sm = nn.Softmax(1)
        for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs = Variable(sample['img'].cuda(), volatile=True)
            if args.TTA:
                bs, ncrops, h, c, w = inputs.size()
                outputs = net(inputs.view(-1, h, w, c))
                probs = sm(outputs).view(bs, ncrops, -1).mean(1).data.cpu().numpy()
            else:
                outputs = net(inputs)
                probs = sm(outputs).data.cpu().numpy()
                
            preds.append(probs)
            gt.append(sample['label'].numpy())
            gc.collect()
        gc.collect()

    preds = np.vstack(preds)
    gt = np.hstack(gt)
    
    np.save('val_predicts/{}_preds.npy'.format(args.experiment), preds)
    np.save('val_predicts/{}_gt.npy'.format(args.experiment), gt)
    
    cm = confusion_matrix(gt, preds.argmax(1))
    print(cm)
    #print('Kappa:', cohen_kappa_score(gt, preds.argmax(1), weights='quadratic'))
    print('Accuracy:', accuracy_score(gt, preds.argmax(1)))
