import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import cv2
import os
import numpy as np
from torch.autograd import Variable
from PIL import Image
import random
import cv2


class mydataset(Dataset):

    def __init__(self, root_dir, data_list, data_dict, mode='train', transforms=None, shape=(224,224)):
        self.root_dir = root_dir
        self.data_dict = data_dict
        self.data_list = data_list
        self.transforms = transforms
        self.mode = mode
        self.shape = shape
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_list[idx])
        img = cv2.imread(img_name,1)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        name = self.data_list[idx].split('/')[2]
        name = name.split('.')[0]
        if self.mode == 'train':
          name = name.split('_')[0]
        else:
          name = name.split('_')[2]  
          name = int(name)
        
        
        if self.transforms:
            img = self.transforms(img)
        
        sample = img, self.data_dict[name]
        return sample

def test(model, testloader, criterion):
    correct = 0
    total = 0
    total_loss = 0
    cnt = 0
    for images, labels in testloader:
        cnt += 1
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        loss = criterion(outputs,labels)
        total_loss += float(loss)
        total += labels.size(0)
        correct += (predicted == labels).sum()
       
    return  100.0 * float(correct) / float(total), float(total_loss) / float(cnt)
