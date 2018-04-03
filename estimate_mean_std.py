from __future__ import print_function
import sys
from tqdm import tqdm
import numpy as np
import argparse
import os
import pandas as pd
from termcolor import colored
import gc
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from dataset import CameraClassificationTrainDataset
from augmentation import * 
from validate_predicts import test_set_transform

def calc_mean_std(args,dataset_transforms, n_rounds=1):

    data_frame = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
    camera_id_train = data_frame['camera'].values
    fname_train = data_frame['fname'].values
    ds_size = len(fname_train)

    ds = CameraClassificationTrainDataset(args.dataset, fname_split=fname_train, classes_split=camera_id_train, transform=dataset_transforms, in_memory=args.in_memory)

    loader = data.DataLoader(ds,
                             batch_size=args.bs,
                             num_workers=args.n_threads,
                             shuffle=True)
    means = []
    stdevs = []
    for n in range(0, n_rounds):
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            pixels = sample['img'].cuda()
            pixels = pixels.view(-1, 3, pixels.size(2)*pixels.size(3))
            means.append(pixels.mean(2).cpu().numpy())
            stdevs.append(pixels.std(2).cpu().numpy())
            

            
    return np.mean(np.vstack(means), 0), np.std(np.vstack(stdevs), 0)

