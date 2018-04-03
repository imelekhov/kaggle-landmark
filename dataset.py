import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
#import jpeg4py as jpeg
import cv2
import os
cv2.ocl.setUseOpenCL(False)
import io
from tqdm import tqdm

import torchvision.transforms as transforms


def get_class_dict():
    camera_classes = ['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x', \
                'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X', \
                'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']
    return camera_classes



class CameraClassificationTrainDataset(data.Dataset):
    def __init__(self, image_path, fname_split, classes_split, transform, mode='train', in_memory=False):
        self.image_path = image_path
        self.fname_split = fname_split
        self.classes_split = classes_split
        self.transform = transform
        self.camera_classes = get_class_dict()
        self.mode = mode
        if in_memory:
            self.in_memory_data = {}
            for ind in tqdm(range(self.fname_split.shape[0])):
                filename = os.path.join(self.image_path, self.mode, self.classes_split[ind], self.fname_split[ind])
                with open(filename, 'rb') as fin:
                    img = fin.read()
                
                self.in_memory_data[filename] = cv2.imread(filename)
        else:
            self.in_memory_data = None
            
        
    def __getitem__(self, ind):
        filename = os.path.join(self.image_path, self.mode, self.classes_split[ind], self.fname_split[ind])
        if self.in_memory_data is None:
            img = cv2.imread(filename)#jpeg.JPEG(filename).decode()
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            #nparr = np.fromstring(self.in_memory_data[filename], np.uint8)
            #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = self.in_memory_data[filename]
        class_id = self.camera_classes.index(self.classes_split[ind])

        img = self.transform(img)
        return {'img': img, 'label': class_id, 'fname':filename}

    def __len__(self):
        return self.fname_split.shape[0]


class CameraClassificationTestDataset(data.Dataset):
    def __init__(self, image_path, fname, transform):
        self.image_path = image_path
        self.fname = fname
        self.transform = transform
        self.camera_classes = get_class_dict()

    def __getitem__(self, ind):
        filename = os.path.join(self.image_path, 'test', self.fname[ind])
        img = cv2.imread(filename)
        img = self.transform(img)
        return {'img': img}

    def __len__(self):
        return self.fname.shape[0]
