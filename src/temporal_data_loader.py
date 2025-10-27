# src/temporal_data_loader.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class TemporalCXRDataset(Dataset):
    def __init__(self, csv_file, img_dir, sequence_length=2, transform=None, max_patients=1000):
        """
        Creates temporal sequences from NIH data
        sequence_length: number of images per sequence (2-3 works best with NIH)
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
        
        # Extract patient ID from image name (format: '00000001_001.png')
        self.df['patient_id'] = self.df['image_id'].str.split('_').str[0]
        
        # Create patient sequences
        self.sequences = self._create_patient_sequences(max_patients)
        
        print(f"Created {len(self.sequences)} temporal sequences")
    
    def _create_patient_sequences(self, max_patients):
        """Create temporal sequences grouped by patient"""
        sequences = []
        patient_counts = {}
        
        # Group by patient and sort by image ID (assumes chronological order)
        for patient_id, group in self.df.groupby('patient_id'):
            if len(sequences) >= max_patients:
                break
                
            patient_images = group.sort_values('image_id').to_dict('records')
            
            if len(patient_images) >= self.sequence_length:
                # Create overlapping sequences for patients with enough images
                for i in range(len(patient_images) - self.sequence_length + 1):
                    sequence = patient_images[i:i + self.sequence_length]
                    sequences.append({
                        'images': sequence,
                        'patient_id': patient_id,
                        'sequence_id': f"{patient_id}_{i}"
                    })
            else:
                # For patients with fewer images, create padded sequences
                sequence = patient_images.copy()
                # Pad with copies of the last image
                while len(sequence) < self.sequence_length:
                    sequence.append(sequence[-1])
                
                sequences.append({
                    'images': sequence,
                    'patient_id': patient_id,
                    'sequence_id': f"{patient_id}_padded"
                })
            
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        
        print(f"Patients used: {len(patient_counts)}")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        
        images = []
        all_labels = []
        
        for img_data in sequence_data['images']:
            # Load image
            img_path = os.path.join(self.img_dir, img_data['path'])
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                # Fallback: create blank image if file not found
                img = Image.new('RGB', (224, 224), color='black')
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
            
            # Get labels for this image
            img_labels = [img_data.get(cls, 0) for cls in self.classes]
            all_labels.append(img_labels)
        
        # Stack images and labels
        images_tensor = torch.stack(images)  # Shape: [sequence_length, 3, H, W]
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)  # Shape: [sequence_length, n_classes]
        
        # For temporal prediction: use last image's labels as target
        # This predicts disease progression from sequence
        return images_tensor, labels_tensor[-1], sequence_data['sequence_id']

def get_temporal_transforms(image_size=224, is_train=True):
    """Transforms for temporal sequences"""
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])