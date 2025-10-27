# src/multi_modal_loader.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class MultiModalCXRDataset(Dataset):
    def __init__(self, csv_file, img_dir, clinical_csv, transform=None, sequence_length=1):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
        
        # Load clinical data
        self.clinical_df = pd.read_csv(clinical_csv, index_col='patient_id')
        
        # Extract patient ID
        self.df['patient_id'] = self.df['image_id'].str.split('_').str[0]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        
        # Load image
        img_path = os.path.join(self.img_dir, row['path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Get labels
        labels = torch.tensor([row[cls] for cls in self.classes], dtype=torch.float32)
        
        # Get clinical data
        if patient_id in self.clinical_df.index:
            clinical_row = self.clinical_df.loc[patient_id]
            clinical_data = torch.tensor([
                clinical_row['age'],
                clinical_row['gender'],
                clinical_row['smoking_history'],
                clinical_row['oxygen_saturation'],
                clinical_row['fever'],
                clinical_row['cough'],
                clinical_row['shortness_of_breath'],
                clinical_row['chest_pain']
            ], dtype=torch.float32)
        else:
            # Default clinical data if not found
            clinical_data = torch.zeros(8, dtype=torch.float32)
        
        return img, clinical_data, labels, row['image_id']