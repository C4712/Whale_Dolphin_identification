from utils.triplet_loss import *
from utils.Whalemodel import model_whale
import os
import gc
import cv2
import math
import copy
import time
import random

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


CONFIG = {"seed": 2022,
          "epochs": 30,
          "img_size": 512,
          "model_name": "tf_efficientnet_b0",
          "num_classes": 15587,
          "train_batch_size": 8,
          "valid_batch_size": 8,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-7,
          "T_max": 30,
          "weight_decay": 1e-5,
          "n_fold": 5,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # ArcFace Hyperparameters
          "s": 30.0, 
          "m": 0.50,
          "ls_eps": 0.0,
          "easy_margin": False,
          "output_path":'./weight/all_data/'
          }

if not os.path.exists(CONFIG['output_path']):
    os.makedirs(CONFIG['output_path'])

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(CONFIG['seed'])

df = pd.read_csv('../data/train_fold.csv')

class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['individual_id_map'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader



def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    train_map5_sum = 0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        batch_size = images.size(0)
        
        global_feat, local_feat, results = model(images)
        model.getLoss(global_feat, local_feat, results, labels)
        loss = model.loss
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
        results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
        top1_batch = accuracy(results, labels, topk=(1,))[0]
        map5_batch = mapk(labels, results, k=5)
        loss = loss.data.cpu().numpy()
        sum += 1
        train_loss_sum += loss
        train_top1_sum += top1_batch
        train_map5_sum += map5_batch
        
        bar.set_postfix(Epoch=epoch, Train_Loss=loss,LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return loss

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0.0
        valid_loss, index_valid= 0, 0
        all_results = []
        all_labels = []

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:        
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)
            global_feat, local_feat, results = model(images)
            model.getLoss(global_feat, local_feat, results, labels)

            results = torch.sigmoid(results)

            all_results.append(results)
            all_labels.append(labels)

            b = len(labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            index_valid += b
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)

        map5s, top1s, top5s = [], [], []
        if 1:
            ts = np.linspace(0.1, 0.9, 9)
            for t in ts:
                results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda() * t], 1)
                top1_, top5_ = accuracy(results_t, all_labels)
                map5_ = mapk(all_labels, results_t, k=5)
                map5s.append(map5_)
                top1s.append(top1_)
                top5s.append(top5_)
            map5 = max(map5s)
            i_max = map5s.index(map5)
            top1 = top1s[i_max]
            top5 = top5s[i_max]
            best_t = ts[i_max]

        valid_loss /= index_valid

    return valid_loss, top1, top5, map5, best_t

def run_training(model, optimizer, scheduler, device, num_epochs):
    best_map5 = 0
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        val_epoch_loss, top1, top5, map5, best_t = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss}) map5：{map5}")
        if map5 > best_map5:
            best_map5 = map5
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"model_{epoch}.pth"
            torch.save(model.state_dict(), CONFIG['output_path']+PATH)
            print(f"Model Saved{sr_}")
        
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history
if __name__ == '__main__':
    train_loader, valid_loader = prepare_loaders(df, fold=0)
    num_classes = 15587
    model = model_whale(num_classes=num_classes, inchannels=3, model_name='senet154').cuda()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=CONFIG['epochs'])
