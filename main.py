import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from dataset import *
from torch.backends import cudnn
import os
import sys
from senetsn import se_resnext50_32x4d
from utils import ColorAugmentation
from datetime import datetime

batch_size = 128
epoch = 100
learning_rate = 0.075
load_model = False
record_result = False
load_path = './save/se_resnext_sn1_acc.pkl'
save_path = 'save'
net_name = 'se_resnext_sn1'
train_transforms = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ColorAugmentation(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229                                                         ,0.224,0.225]),
                                   ])
valid_transforms = transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,                                                         0.224,0.225])
                                  ])
root_dir = '/data-173/imagenet'
train_dict = torch.load('list/train_dict.pkl')
valid_dict = torch.load('list/valid_dict.pkl')
train_list = torch.load('list/train_list.pkl')
valid_list = torch.load('list/valid_list.pkl')
trainset = mydataset(root_dir=root_dir, data_list=train_list, data_dict=train_dict,transforms=train_transforms)
validset = mydataset(root_dir=root_dir, data_list=valid_list, data_dict=valid_dict, mode='valid',transforms=valid_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(validset, batch_size=64, 
                                         shuffle=False, num_workers=4)

net = se_resnext50_32x4d(pretrained=None)

if load_model:
  net.load_state_dict(torch.load(load_path))

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

net = nn.DataParallel(net,device_ids=[0,1,2,3])
criterion  = torch.nn.CrossEntropyLoss()
cudnn.benchmark = True
net = net.cuda()
criterion = criterion.cuda()
 
best_acc = 0

for step in range(epoch):
    pre_time = datetime.now() 
    epoch_loss = 0.0
    running_loss = 0.0
    net.train()
    total = 0
    scheduler.step()
    for i, data in enumerate(trainloader, 0): 
        total += 1 
        inputs, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad() 
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward() 
        optimizer.step() 
        running_loss += float(loss)
        epoch_loss += float(loss)
        if i % 1000 == 999:
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60) 
            print('[%d, %5d] loss: %.3f {:.0f}:{:.0f}:{:.0f}' % (step+1, i+1, running_loss / 1000, h, m, s))
            pre_time = datetime.now()
            running_loss = 0.0
    if step % 1 == 0:
        pre_time = datetime.now()
        net.eval()
        test_acc,test_loss = test(net,testloader,criterion)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
         
        print('test_accuracy and loss in epoch %d : %.3f %.3f {:.0f}:{:.0f}:{:.0f}'%(step,test_acc,test_loss, h, m, s))
        #print('epoch_loss in epoch %d : %.3f'%(step,epoch_loss/total))     
        if best_acc <= test_acc:
            best_acc = test_acc
            torch.save(net.state_dict(),os.path.join(save_path,net_name+'_acc.pkl'))
print('Finished training') 

