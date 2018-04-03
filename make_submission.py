from __future__ import print_function


from dataset import CameraClassificationTestDataset, get_class_dict
from train_utils import train_epoch
from val_utils import validate_epoch
from augmentation import *
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score

import time
import pickle

from collections import OrderedDict

import time
cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transfroms

import model_zoo

print('Torch Version', torch.__version__)
def find_checkpoint(path, pattern='*.pth'):
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return name
    return None

def get_new_dict(dict_with_module):
    new_state_dict = OrderedDict()
    for k, v in dict_with_module.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='data')
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--snapshots_path',  default='snapshots')
parser.add_argument('--snapshot',  default='')
parser.add_argument('--submission_path',  default='./submissions/')
parser.add_argument('--bs', type=int, default=8)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--seed', type=int, default=422345)
args = parser.parse_args()


os.makedirs(args.submission_path, exist_ok=True)
os.makedirs(os.path.join(args.submission_path, 'raw_predicts'), exist_ok=True)

tmp = np.load(os.path.join(args.snapshots_path, 'mean_std.npy'))
mean_vector, std_vector = tmp
normTransform = transforms.Normalize(torch.from_numpy(mean_vector), torch.from_numpy(std_vector))



with open(os.path.join(args.snapshots_path, args.snapshot, 'args.pkl'), 'rb') as f:
    tmp = pickle.load(f)
    args.experiment = tmp.experiment
    args.crop_size = tmp.crop_size

test_transforms = transforms.Compose([
  lambda x: center_crop(x, args.crop_size),
  transforms.ToTensor(),
  normTransform
])


# Preparing the dataset
data_frame = pd.read_csv(os.path.join(args.dataset, 'sample_submission.csv'))
fname_test = data_frame['fname']
results = 0
for fold in range(args.n_folds):
    print(colored('==> ', 'red')+'fold: ', fold)

    path = os.path.join(args.snapshots_path, args.snapshot, 'fold_'+str(fold))
    checkpoint_name = find_checkpoint(path)
    checkpoint = torch.load(os.path.join(path, checkpoint_name))
    
    
    net = model_zoo.get_model(args)
    net.load_state_dict(checkpoint)
    print(colored('==> ', 'green')+'Loaded weights: ', args.experiment)
    net = nn.DataParallel(net)
    net.cuda()
    net.eval()

    test_ds = CameraClassificationTestDataset(args.dataset, fname=fname_test, transform=test_transforms)
    test_loader = data.DataLoader(test_ds,
                                    batch_size=args.bs,
                                    num_workers=args.n_threads,
                                    sampler=data.sampler.SequentialSampler(test_ds)
                                 )
    preds = []
    sm = nn.Softmax(1)
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs = Variable(sample['img'].cuda(), volatile=True)
        outputs = net(inputs)
        probs = sm(outputs).data.cpu().numpy()
        preds.append(probs)
        gc.collect()
    gc.collect()
    results += np.vstack(preds)

results /= args.n_folds

np.save(os.path.join(args.submission_path, 
                     'raw_predicts', 
                     '{}_{}_{}.npy'.format(args.experiment, args.snapshot, args.n_folds)),
                    results)
camera_classes = np.array(get_class_dict())
prediction_camera_class = camera_classes[results.argmax(1)]
data_frame['camera'] = prediction_camera_class
data_frame.to_csv(os.path.join(args.submission_path, '{}_{}_{}.csv'.format(args.experiment, args.snapshot, args.n_folds)), index=False)
