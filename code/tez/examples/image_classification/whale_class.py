import sys
sys.path.append('../../../tez/')
sys.path.append('../../../Humpback-Whale-Identification-1st/')
from models import *
from dataSet import *
from utils import *
import argparse
import os

import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from torch.nn import functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

INPUT_PATH = "../../../../data/"
IMAGE_PATH = "../../../../data/train_images-128-128/"
MODEL_PATH = "./weight/tez"
MODEL_NAME = 'whale'
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
EPOCHS = 20
IMAGE_SIZE = 128
num_classes = 15587

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

model_backbone = model_whale(num_classes=num_classes, inchannels=3, model_name='senet154').cuda()

def get_train_file_path(id):
    return f"{IMAGE_PATH}/{id}"

df_data = pd.read_csv('../../../../data/train.csv')
df_data['file_path'] = df_data['image'].apply(get_train_file_path)
encoder = LabelEncoder()
df_data['individual_id_map'] = encoder.fit_transform(df_data['individual_id'])

skf = StratifiedKFold(n_splits=5)

for fold, ( _, val_) in enumerate(skf.split(X=df_data, y=df_data.individual_id_map)):
      df_data.loc[val_ , "kfold"] = fold

df_data.to_csv('../../../../data/train_folds.csv',index=False)

train_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        albumentations.CoarseDropout(p=0.5),
        albumentations.Cutout(p=0.5),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



class Whalemodel(tez.Model):    
    def __init__(self):
        super().__init__()

        self.effnet = model_backbone
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}

        all_results = torch.cat([outputs], 0)
        all_labels = torch.cat([targets], 0)
        map5s = []
        if 1:
            ts = np.linspace(0.1, 0.9, 9)
            for t in ts:
                results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda() * t], 1)
                map5_ = mapk(all_labels, results_t, k=5)
                map5s.append(map5_)
            map5 = max(map5s)
#             i_max = map5s.index(map5)
            
#         accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": map5}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape
        global_feat, local_feat, results = self.effnet(image)
        if targets is not None:
            self.effnet.getLoss(global_feat, local_feat, results, targets)
            results = torch.sigmoid(results)
            metrics = self.monitor_metrics(results, targets)
            return results, self.effnet.loss, metrics
        return results, None, None

if __name__ == "__main__":
    current_fold = 0
    dfx = pd.read_csv(os.path.join(INPUT_PATH, "train_folds.csv"))
    df_train = dfx[dfx.kfold != current_fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == current_fold].reset_index(drop=True)
    train_image_paths = [os.path.join(IMAGE_PATH, x) for x in df_train.image.values]
    valid_image_paths = [os.path.join(IMAGE_PATH, x) for x in df_valid.image.values]
    train_targets = df_train.individual_id_map.values
    valid_targets = df_valid.individual_id_map.values

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=train_aug,
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets,
        augmentations=valid_aug,
    )
    model = Whalemodel()
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + f"_fold_{current_fold}.bin"),
        patience=3,
        mode="min",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=TRAIN_BATCH_SIZE,
        valid_bs=VALID_BATCH_SIZE,
        device="cuda",
        epochs=EPOCHS,
        callbacks=[es],
        fp16=True,
    )
    model.save(os.path.join(MODEL_PATH, MODEL_NAME + f"_fold_{current_fold}.bin"))

