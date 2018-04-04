import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import gc


def train_epoch(epoch, net, optimizer, train_loader, criterion, max_ep):

    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, sample in pbar:
        optimizer.zero_grad()
        # forward + backward + optimize
        labels = Variable(sample['label'].long().cuda(async=True))
        inputs = Variable(sample['img'].cuda(), requires_grad=True)
        
        outputs = net(inputs)
        print(outputs.shape)

        loss = criterion(outputs, labels)
        print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
        
        loss.backward()
        optimizer.step()

        running_loss += loss.data.cpu().numpy()[0]
        pbar.set_description('Running_loss: %.3f / total_loss: %.3f' % (running_loss / (i+1), loss.data.cpu().numpy()[0]))

        gc.collect()
    gc.collect()

    return running_loss/n_batches


