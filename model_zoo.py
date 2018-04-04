import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import torchvision.models as models
from copy import deepcopy

import sys
sys.path.append('pretrained-models.pytorch')
import pretrainedmodels

def get_model(args):
    net = None
    if 'resnet' in args.experiment:
        for resnet_conf in [18, 34, 50, 101, 152]:
            if args.experiment == 'resnet'+str(resnet_conf):
                net = ResNet(resnet_conf, args.drop, 3)
    elif 'inception' in args.experiment:
        for incept_conf in ['V3', 'V4']:
            if args.experiment == 'inception'+str(incept_conf):
                net = InceptionV4(args.drop, 10)
    elif 'densenet' in args.experiment:
        for densenet_conf in [121, 169, 201]:
            if args.experiment == 'densenet'+str(densenet_conf):
                net = DenseNet(densenet_conf, args.drop, 10)
    elif 'nas' in args.experiment:
        net = NASNet(args.drop, 10)
    elif 'vgg16' in args.experiment:
        net = VGG16(ncls=10, drop=args.drop)
    elif 'tiny' in args.experiment:
        net = TinyCNN(ncls=10, drop=args.drop)
    return net


class InceptionV4(nn.Module):
    def __init__(self, drop, ncls):
        super().__init__()
        model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        self.encoder = list(model.features.children())[:-1]
        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)
        if drop > 0:
            self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(1536, ncls))
        else:
            self.classifier = nn.Linear(1536, ncls)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module): #vgg16_bn
    def __init__(self, drop, ncls):
        super().__init__()
        model = models.vgg16_bn(pretrained=True)
        self.encoder = list(model.features.children())[:-1]
        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)
        if drop > 0:
            self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(512, 1024), nn.ReLU(inplace=True), \
                                            nn.Dropout(drop), nn.Linear(1024, 1024), nn.ReLU(inplace=True), \
                                            nn.Linear(1024, ncls))
        else:
            self.classifier = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(inplace=True), \
                                            nn.Linear(1024, 1024), nn.ReLU(inplace=True), \
                                            nn.Linear(1024, ncls))


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, layers, drop, ncls):
        super().__init__()

        if layers == 121:
            model = models.densenet121(pretrained=True)
            nfeats = 1024
        if layers == 169:
            model = models.densenet169(pretrained=True)
            nfeats = 1664
        if layers == 201:
            model = models.densenet201(pretrained=True)
            nfeats = 1920

        self.encoder = list(model.children())[:-1]
        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(nfeats, ncls))
        else:
            self.classifier = nn.Linear(nfeats, ncls)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class ResNet(nn.Module):
    def __init__(self, layers, drop, ncls):
        super().__init__()

        if layers == 18:
            model = models.resnet18(pretrained=True)
        if layers == 34:
            model = models.resnet34(pretrained=True)
            print('we are here!!!!')
        if layers == 50:
            model = models.resnet50(pretrained=True)
        if layers == 101:
            model = models.resnet101(pretrained=True)
        if layers == 152:
            model = models.resnet152(pretrained=True)
        self.encoder = list(model.children())[:-2]
        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)
        nfeats = 2048
        if layers < 50:
            nfeats = 1024

        if drop > 0:
            self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(nfeats, ncls))
        else:
            self.classifier = nn.Linear(nfeats, ncls)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, drop, ncls):
        super().__init__()
        model = models.densenet201(pretrained=True)
        #self.encoder = nn.Sequential(*list(model.features.children()))
        self.encoder = list(model.features.children())
        # getting the number of channels from the last conv layer
        #num_chnls = self.encoder[40].out_channels
        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)
        if drop > 0:
            self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(1920, ncls))
        else:
            self.classifier = nn.Linear(1920, ncls)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TinyCNN(nn.Module):
    def __init__(self, ncls, drop):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=(9,9), stride=2),
          nn.BatchNorm2d(32),
          nn.PReLU(),
          nn.Conv2d(32, 64, kernel_size=(7,7), stride=2),
          nn.BatchNorm2d(64),
          nn.PReLU(),
          nn.Conv2d(64, 64, kernel_size=(5,5), stride=2),
          nn.BatchNorm2d(64),
          nn.PReLU(),
          nn.Conv2d(64, 128, kernel_size=(3,3), stride=2),
          nn.BatchNorm2d(128),
          nn.PReLU(),
          nn.Conv2d(128, 128, kernel_size=(3,3), stride=2),
          nn.BatchNorm2d(128),
          nn.PReLU(),
          nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(drop),
                                        nn.Linear(128, 1024), 
                                        nn.PReLU(),
                                        nn.Dropout(drop),
                                        nn.Linear(1024, 1024),
                                        nn.PReLU(),
                                        nn.Dropout(drop),
                                        nn.Linear(1024, ncls)
                                       )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
