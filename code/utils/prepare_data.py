import cv2
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import joblib

ROOT_DIR = '../../data/happy-whale-and-dolphin'
TRAIN_DIR = '../data/happy-whale-and-dolphin/train_images/'
TEST_DIR = '../data/happy-whale-and-dolphin/test_images/'

def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id}"

df = pd.read_csv(f"{ROOT_DIR}/train.csv")
df['file_path'] = df['image'].apply(get_train_file_path)

encoder = LabelEncoder()
df['individual_id_map'] = encoder.fit_transform(df['individual_id'])

with open("../le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)
# encoder.inverse_transform([1636])

skf = StratifiedKFold(n_splits=5)
for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.individual_id_map)):
      df.loc[val_ , "kfold"] = fold
df.to_csv('../../data/train_fold.csv')