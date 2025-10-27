# src/data_loader.py
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CXRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_cols=None):
        """
        csv_file: csv with columns ['image_id','path', <label_cols>...]
        label_cols: list of label column names (multi-label)
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_cols = label_cols or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.label_cols:
            labels = torch.tensor(row[self.label_cols].values.astype('float32'))
            return img, labels, row['image_id']
        else:
            return img, row['image_id']

def get_transforms(image_size=224, is_train=True):
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
