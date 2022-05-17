import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time
import matplotlib.pyplot as plt


from models import EfficientNet_b4
from models import EfficientNet_b0

device = torch.device('cuda')

class Network_b3(nn.Module):
    def __init__(self,mode = 'train'):
        super(Network_b3, self).__init__()
        self.mode = mode
        if self.mode == 'train':
          self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=88, drop_path_rate = 0.2)
        if self.mode == 'test':
          self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=88, drop_path_rate = 0)
        
    def forward(self, x):
        x = self.model(x)
        return x

model = EfficientNet_b4()
model_b3 = Network_b3().to(device)
# load
#model.load_state_dict(torch.load("/home/fds/Dev/Python/pytorch/dacon/model/tmp/effi-b04_model_10_f1_socore_0.764871609155833.pth"))


model.to(device)


model.eval()
f_pred = []

test_png = sorted(glob('/home/fds/Dev/Python/pytorch/dacon/open/test/*.png'))
def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

test_imgs = [img_load(n) for n in tqdm(test_png)]
np.save('/home/fds/Dev/Python/pytorch/dacon/open/' + 'test_imgs_384', np.array(test_imgs))
test_imgs = np.load('/home/fds/Dev/Python/pytorch/dacon/open/' + 'test_imgs_384.npy')


train_y = pd.read_csv("/home/fds/Dev/Python/pytorch/dacon/open/train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
          train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.433038, 0.403458, 0.394151],
                                     std = [0.181572, 0.174035, 0.163234]),
                transforms.RandomAffine((-45, 45)),
                
            ])
          img = train_transform(img)
        if self.mode == 'test':
          test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.418256, 0.393101, 0.386632],
                                     std = [0.195055, 0.190053, 0.185323])
            ])
          img = test_transform(img)

        
        label = self.labels[idx]
        return img, label
    


pred_ensemble = []
batch_size = 32
# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16)

model_b3_path_list = [
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/best_model_0.pth',
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/best_model_1.pth',
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/best_model_2.pth',
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/best_model_3.pth',
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/best_model_4.pth',
]
for i in range(len(model_b3_path_list)):
  model_test = model_b3
  model_test.load_state_dict(torch.load(model_b3_path_list[i])['state_dict'])
  model_test.eval()
  pred_prob = []
  with torch.no_grad():
      for batch in (test_loader):
          x = torch.tensor(batch[0], dtype = torch.float32, device = device)
          with torch.cuda.amp.autocast():
              pred = model_test(x)
              pred_prob.extend(pred.detach().cpu().numpy())
      pred_ensemble.append(pred_prob)

model_path_list = [
    '/home/fds/Dev/Python/pytorch/dacon/model/ensemble/effi-b04_Kfold_model_1_f1_4_1.0.pth',

]
for i in range(len(model_path_list)):
  model_test = model
  model_test.load_state_dict(torch.load(model_path_list[i]))
  model_test.eval()
  pred_prob = []
  with torch.no_grad():
      for batch in (test_loader):
          x = torch.tensor(batch[0], dtype = torch.float32, device = device)
          with torch.cuda.amp.autocast():
              pred = model_test(x)
              pred_prob.extend(pred.detach().cpu().numpy())
      pred_ensemble.append(pred_prob)

print(len(pred_ensemble))

e_pred = np.zeros(np.array(pred_ensemble[0]).shape)
for pre in pred_ensemble:
    e_pred += np.array(pre)
f_pred = np.array(e_pred).argmax(1).tolist()

label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv("/home/fds/Dev/Python/pytorch/dacon/open/sample_submission.csv")

submission["label"] = f_result

submission.to_csv("/home/fds/Dev/Python/pytorch/dacon/open/baseline.csv", index = False)