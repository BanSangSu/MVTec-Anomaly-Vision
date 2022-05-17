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

import copy


device = torch.device('cuda')

train_png = sorted(glob('/home/fds/Dev/Python/pytorch/dacon/open/train/*.png'))

train_y = pd.read_csv("/home/fds/Dev/Python/pytorch/dacon/open/train_df.csv")

train_labels = train_y["label"]
path = "/home/fds/Dev/Python/pytorch/dacon/"

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

train_imgs = [img_load(m) for m in tqdm(train_png)]
np.save(path + 'train_imgs_384', np.array(train_imgs))
train_imgs = np.load(path + 'train_imgs_384.npy')

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

# n_train_imgs = len(train_imgs)//100*90
# # Train
# train_dataset = Custom_dataset(np.array(train_imgs[:n_train_imgs]), np.array(train_labels[:n_train_imgs]), mode='train')
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)

# # validation
# vaild_dataset = Custom_dataset(np.array(train_imgs[n_train_imgs:]), np.array(train_labels[n_train_imgs:]), mode='test')
# vaild_loader = DataLoader(vaild_dataset, shuffle=False, batch_size=batch_size, num_workers=4)



best_loss = 0.29704457932505113
best_f1 = 0

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


# model = EfficientNet_b0()
# # model = Network().to(device)
# # load
# #model.load_state_dict(torch.load("/home/fds/Dev/Python/pytorch/dacon/model/tmp/effi-b0_model_10_f1_socore_0.764871609155833.pth"))

# model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False)
# criterion = nn.CrossEntropyLoss()
# scaler = torch.cuda.amp.GradScaler() 
    

train_losses = []
vaild_losses = []
best=0
# for epoch in range(epochs):
#     start=time.time()
#     train_loss = 0
#     train_pred=[]
#     train_y=[]
#     model.train()
#     for batch in (train_loader):
#         optimizer.zero_grad()
#         x = torch.tensor(batch[0], dtype=torch.float32, device=device)
#         y = torch.tensor(batch[1], dtype=torch.long, device=device)
#         with torch.cuda.amp.autocast():
#             pred = model(x)
#         loss = criterion(pred, y)


#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         train_loss += loss.item()/len(train_loader)
#         train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
#         train_y += y.detach().cpu().numpy().tolist()
        
    
#     train_f1 = score_function(train_y, train_pred)

#     TIME = time.time() - start
#     print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
#     print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')


#     vaild_loss = 0
#     vaild_pred=[]
#     vaild_y=[]
#     model.eval()
#     with torch.no_grad():
#         for batch in (vaild_loader):
#             x = torch.tensor(batch[0], dtype=torch.float32, device=device)
#             y = torch.tensor(batch[1], dtype=torch.long, device=device)
#             with torch.cuda.amp.autocast():
#                 pred = model(x)
#             loss = criterion(pred, y)
            
#             vaild_loss += loss.item()/len(vaild_loader)
#             vaild_pred += pred.argmax(1).detach().cpu().numpy().tolist()
#             vaild_y += y.detach().cpu().numpy().tolist()
            
        
#         vaild_f1 = score_function(vaild_y, vaild_pred)

#         TIME_E = time.time() - start
#         print(f'epoch : {epoch+1}/{epochs}    time : {TIME_E:.0f}s/{TIME_E*(epochs-epoch-1):.0f}s')
#         print(f'Vaild    loss : {vaild_loss:.5f}    f1 : {vaild_f1:.5f}')

#     # train_losses.append(train_loss)
#     # vaild_losses.append(vaild_loss)
#     # plt.plot(vaild_losses, label='val_loss')
#     # plt.plot(train_losses, label='train_loss')
#     # plt.xlabel('effi-b04_00')
#     # plt.ylabel('loss')
#     # plt.legend()
#     # plt.show()

#     savepath = '/home/fds/Dev/Python/pytorch/dacon/model/tmp/effi-b00_model_{}_{}_{}.pth'

#     # if best_loss > vaild_loss:
#     #     print("-------- SAVE MODEL --------")
#     #     best_loss = vaild_loss
#     #     best_model_wts = copy.deepcopy(model.state_dict())
#     #     torch.save(model.state_dict(), savepath.format(epoch, "model",best_loss))
#     if best_f1 < vaild_f1:
#         print("-------- SAVE MODEL --------")
#         best_f1 = vaild_f1
#         best_model_wts = copy.deepcopy(model.state_dict())
#         torch.save(model.state_dict(), savepath.format(epoch, "f1", best_f1))


from sklearn.model_selection import StratifiedKFold
import gc

cv = StratifiedKFold(n_splits = 5, random_state = 2022,shuffle=True)
batch_size = 16
epochs = 50
pred_ensemble = []

model = EfficientNet_b4()
# load
model.to(device)

for idx, (train_idx, val_idx) in enumerate(cv.split(train_imgs, np.array(train_labels))):
  print("----------fold_{} start!----------".format(idx))
  t_imgs, val_imgs = train_imgs[train_idx],  train_imgs[val_idx]
  t_labels, val_labels = np.array(train_labels)[train_idx], np.array(train_labels)[val_idx]

  # Train
  train_dataset = Custom_dataset(np.array(t_imgs), np.array(t_labels), mode='train')
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

  # Val
  val_dataset = Custom_dataset(np.array(val_imgs), np.array(val_labels), mode='test')
  val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

  gc.collect()
  torch.cuda.empty_cache()
  best=0

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False)
  criterion = nn.CrossEntropyLoss()
  scaler = torch.cuda.amp.GradScaler()  

  best_f1 = 0
  early_stopping = 0
  for epoch in range(epochs):
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
    train_f1 = score_function(train_y, train_pred)
    state_dict= model.state_dict()
    model.eval()
    with torch.no_grad():
      val_loss = 0 
      val_pred = []
      val_y = []
      

      for batch in (val_loader):
        x_val = torch.tensor(batch[0], dtype = torch.float32, device = device)
        y_val = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred_val = model(x_val)
        loss_val = criterion(pred_val, y_val)

        val_loss += loss_val.item()/len(val_loader)
        val_pred += pred_val.argmax(1).detach().cpu().numpy().tolist()
        val_y += y_val.detach().cpu().numpy().tolist()
      val_f1 = score_function(val_y, val_pred)

      if val_f1 > best_f1:
        best_epoch = epoch
        best_loss = val_loss
        best_f1 = val_f1
        early_stopping = 0

        savepath = '/home/fds/Dev/Python/pytorch/dacon/model/tmp/effi-b04_Kfold_model_{}_{}_{}_{}.pth'
        torch.save(model.state_dict(), savepath.format(epoch, "f1",idx,best_f1))
        print('-----------------Best Model SAVE:{} epoch----------------'.format(best_epoch+1))
      else:
          early_stopping += 1

            # Early Stopping
      if early_stopping == 20:
        TIME = time.time() - start
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
        print(f'Val    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
        break

    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    print(f'Val    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
