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


class DistractorDataset(data.Dataset):
    def __init__(self, image_path, csv_data_file, transform):
        self.image_path = image_path
        self.data_frame = pd.read_csv(csv_data_file)
        self.transform = transform

    def __getitem__(self, ind):
        fname = self.data_frame.values[ind, 0]
        label = self.data_frame.values[ind, 1]
        if (label == 0): # 0 - landmark class
            fname += '.jpg'

        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_path, fname)), cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return {'img': img, 'label': label, 'fname':fname}

    def __len__(self):
        return self.data_frame.shape[0]