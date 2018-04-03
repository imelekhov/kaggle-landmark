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

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, hamming_loss
from visdom import Visdom

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
from train_utils import train_epoch
from val_utils import validate_epoch
import model_zoo
from augmentation import * 
from estimate_mean_std import calc_mean_std
import cv2
from tensorboardX import SummaryWriter
cv2.ocl.setUseOpenCL(False)

cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='data/')
    parser.add_argument('--snapshots',  default='snapshots/')
    parser.add_argument('--experiment',  default='resnet50')
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--logs', default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--in_memory', type=bool, default=False)
    parser.add_argument('--unfreeze_epoch', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--val_bs', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=60)
    parser.add_argument('--n_batches', type=int, default=-1)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--seed', type=int, default=445)
    args = parser.parse_args()
    cur_lr = args.lr

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')
    if not os.path.isdir(os.path.join(args.snapshots, cur_snapshot)):
        os.mkdir(os.path.join(args.snapshots, cur_snapshot))

    with open(os.path.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
        
    img_transforms  = transforms.Compose([
        lambda x: rotate_90(x),
        lambda x: test_set_transform(x),
        lambda x: augment_random_flip(x),
        lambda x: augment_random_crop(x, int(args.crop_size)),
        transforms.ToTensor(),
    ])   
    if os.path.isfile(os.path.join(args.snapshots, 'mean_std.npy')):
        tmp = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        mean_vector, std_vector = calc_mean_std(args, img_transforms)
        np.save(os.path.join(args.snapshots, 'mean_std.npy'), [mean_vector, std_vector])
        
    print(colored('==> ', 'green')+'Mean: ', mean_vector)
    print(colored('==> ', 'green')+'Std: ', std_vector)

    
    normTransform = transforms.Normalize(torch.from_numpy(mean_vector), torch.from_numpy(std_vector))

    
    train_transforms = transforms.Compose([
        img_transforms,
        normTransform
    ])

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

    skf = StratifiedKFold(n_splits=5, random_state=args.seed)

    print(colored('==> ', 'blue')+'Initialized the datasplits....')
    for fold_id, (train_index, test_index) in enumerate(skf.split(fname_train_shuffled, camera_id_train)):
        if args.fold != -1 and fold_id != args.fold:
            continue
        X, X_val = fname_train_shuffled[train_index], fname_train_shuffled[test_index]
        Y, Y_val = camera_id_train_shuffled[train_index], camera_id_train_shuffled[test_index]

        train_ds = CameraClassificationTrainDataset(args.dataset, fname_split=X, classes_split=Y, transform=train_transforms, in_memory=args.in_memory)

        train_loader = data.DataLoader(train_ds,
                                           batch_size=args.bs,
                                           num_workers=args.n_threads,
                                           shuffle=True
                                          )
        # Validation set
        val_ds = CameraClassificationTrainDataset(args.dataset, fname_split=X_val, classes_split=Y_val, transform=val_transforms, in_memory=args.in_memory)

        val_loader = data.DataLoader(val_ds,
                                     batch_size=args.val_bs,
                                     num_workers=args.n_threads,
                                    )

        print(colored('==> ', 'blue')+'Initialized the loaders....')

        net = model_zoo.get_model(args)
        net = nn.DataParallel(net)
        net.cuda()

        optimizer = optim.Adam(net.module.classifier.parameters(),lr=args.lr, weight_decay=args.wd)

        # Scheduler
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
        # Criterion
        criterion = nn.CrossEntropyLoss().cuda()
        

        train_losses = []
        val_losses = []
        val_acc = []
        prev_model = None

        train_started = time.time()
        np.random.seed(args.seed)
        writer = SummaryWriter(os.path.join(args.logs, 'ieee_camera', 'fold_{}'.format(fold_id), cur_snapshot))
        for epoch in range(args.n_epoch):
            if epoch == args.unfreeze_epoch:
                optimizer = optim.Adam(net.parameters(),lr=args.lr, weight_decay=args.wd)
                print(colored('==> ', 'green')+'Layers were unfrozen!....')
            print(colored('==> ', 'blue')+'Epoch:', epoch+1, fold_id, cur_snapshot)
            # Adjusting learning rate using the scheduler
            scheduler.step()
            print(colored('==> ', 'red')+'LR:', scheduler.get_lr())
            # Training one epoch and measure the time
            start = time.time()
            train_loss = train_epoch(epoch, net, optimizer, train_loader, criterion, args.n_epoch)
            epoch_time = np.round(time.time() - start,4)
            print(colored('==> ', 'green')+'Train loss:', train_loss)
            print(colored('==> ', 'green')+'Epoch training time: {} s.'.format(epoch_time))
            # If it is time to start the validation, we will do it
            # args.args.start_val can be used to avoid time-consuming validation
            # in the beginning of the training
            if epoch >= args.start_val:
                start = time.time()
                val_loss, acc = validate_epoch(net, val_loader, criterion)
                val_time = np.round(time.time() - start, 4)
                #Displaying the results
                print(colored('==> ', 'green')+'Val loss:', val_loss)
                print(colored('==> ', 'green')+'Val accuracy:', acc)
                print(colored('==> ', 'green')+'Epoch val time: {} s.'.format(val_time))
                # Storing the logs
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_acc.append(acc)
                writer.add_scalars('Losses_camera', {'train':train_loss, 'val':val_loss}, epoch)
                writer.add_scalars('Accuracy_camera', {'val':acc}, epoch)
            if not os.path.isdir(os.path.join(args.snapshots, cur_snapshot, 'fold_'+str(fold_id))):
                os.mkdir(os.path.join(args.snapshots, cur_snapshot, 'fold_'+str(fold_id)))

            # Making logs backup
            np.save(os.path.join(args.snapshots, cur_snapshot, 'fold_'+str(fold_id), 'logs.npy'), 
                    [train_losses,val_losses,val_acc])

            if epoch > args.start_val:
                # We will be saving only the snapshot which has lowest loss value on the validation set
                cur_snapshot_name = os.path.join(args.snapshots, cur_snapshot, 'fold_'+str(fold_id), 'epoch_{}.pth'.format(epoch+1))
                if prev_model is None:
                    torch.save(net.module.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name
                    best_val= val_loss
                else:
                    if val_loss < best_val:
                        os.remove(prev_model)
                        best_val= val_loss
                        print('Saved snapshot:',cur_snapshot_name)
                        torch.save(net.module.state_dict(), cur_snapshot_name)
                        prev_model = cur_snapshot_name
            
            gc.collect()

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
