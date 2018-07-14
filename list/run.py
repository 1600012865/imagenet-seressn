import torch
import cv2
import os
import numpy as np
root = '/data-173/imagenet'
l = torch.load('train_list.pkl')
res = np.zeros([3])
minh = 9999
minw = 9999
cnt = 0
total = 0
for name in l:
  path = os.path.join(root,name)
  img = cv2.imread(path,1)
  if minh > img.shape[0]:
    minh = img.shape[0]
    n1 = name
  if minw > img.shape[1]:
    minw = img.shape[1]
    n2 = name
  total += img.shape[0] * img.shape[1]
  tmp = np.sum(img,axis=(0,1))
  res += tmp
  cnt += 1
  

print(n1,n2)
print(minh,minw)
print(res/total)
  
