# src/complete_pipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from multi_modal_loader import MultiModalCXRDataset
from multi_modal_model import MultiModalFusionModel, TemporalMultiModalModel
from clinical_data import ClinicalDataGenerator
from temporal_data_loader import get_temporal_transforms

class CompleteLungAISystem:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
        
    def train_multi_modal(self, train_loader, val_loader, epochs=20):
        """Train multi-modal fusion model"""
        print("🚀 Training Multi-Modal Fusion Model...")
        
        model = MultiModalFusionModel(num_classes=len(self.class_names)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        best_auc = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for images, clinical, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, attn_weights, _ = model(images, clinical)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, clinical, labels, _ in val_loader:
                    images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                    outputs, _, _ = model(images, clinical)
                    val_preds.append(torch.sigmoid(outputs).cpu())
                    val_labels.append(labels.cpu())
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score
            y_true = torch.cat(val_labels).numpy()
            y_pred = torch.cat(val_preds).numpy()
            auc = roc_auc_score(y_true, y_pred, average='macro')
            
            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "../models/multi_modal_fusion.pth")
                print(f"✅ Best multi-modal model saved! AUC: {auc:.4f}")
        
        self.models['multi_modal'] = model
        return model
    
    def predict_complete(self, image_path, clinical_data=None, previous_scans=None):
        """Complete multi-modal prediction"""
        # Load image
        from PIL import Image
        transform = get_temporal_transforms(is_train=False)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Prepare clinical data
        if clinical_data is None:
            clinical_data = torch.zeros(1, 8).to(self.device)  # Default clinical data
        else:
            clinical_data = torch.tensor(clinical_data).unsqueeze(0).to(self.device)
        
        # Multi-modal prediction
        if 'multi_modal' in self.models:
            model = self.models['multi_modal']
        else:
            model = MultiModalFusionModel().to(self.device)
            model.load_state_dict(torch.load("../models/multi_modal_fusion.pth", map_location=self.device))
        
        with torch.no_grad():
            outputs, attn_weights, features = model(image_tensor, clinical_data)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Generate explanation
        explanation = self.generate_explanation(image_tensor, clinical_data, attn_weights)
        
        # Temporal analysis if previous scans available
        temporal_analysis = None
        if previous_scans and len(previous_scans) > 0:
            temporal_analysis = self.analyze_temporal_progression(previous_scans + [image_path], clinical_data)
        
        return {
            'predictions': {cls: prob for cls, prob in zip(self.class_names, probs)},
            'attention_weights': attn_weights.squeeze().cpu().numpy(),
            'explanation': explanation,
            'temporal_analysis': temporal_analysis,
            'features': features.cpu().numpy()
        }
    
    def generate_explanation(self, image, clinical, attn_weights):
        """Generate multi-modal explanation"""
        image_importance = attn_weights[0, 0].item()
        clinical_importance = attn_weights[0, 1].item()
        
        explanation = {
            'image_contribution': image_importance,
            'clinical_contribution': clinical_importance,
            'key_factors': []
        }
        
        # Add clinical factor analysis
        clinical_features = ['age', 'gender', 'smoking', 'oxygen', 'fever', 'cough', 'breathlessness', 'chest_pain']
        if clinical_importance > 0.3:  # If clinical data is important
            clinical_values = clinical.squeeze().cpu().numpy()
            for i, (feature, value) in enumerate(zip(clinical_features, clinical_values)):
                if value > 0.5:  # If feature is present/high
                    explanation['key_factors'].append(f"{feature}: {value:.2f}")
        
        return explanation
    
    def analyze_temporal_progression(self, scan_paths, clinical_data):
        """Analyze disease progression over multiple scans"""
        if len(scan_paths) < 2:
            return "Insufficient scans for temporal analysis"
        
        # Load your temporal model
        from temporal_model import TemporalLSTMModel
        temporal_model = TemporalLSTMModel(sequence_length=len(scan_paths)).to(self.device)
        temporal_model.load_state_dict(torch.load("../models/temporal_lstm_model.pth", map_location=self.device))
        
        # Prepare sequence
        transform = get_temporal_transforms(is_train=False)
        images = []
        for path in scan_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
        
        image_sequence = torch.stack(images).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_pred, progress_pred = temporal_model(image_sequence)
            current_probs = torch.sigmoid(current_pred).squeeze().cpu().numpy()
            progress_probs = torch.sigmoid(progress_pred).squeeze().cpu().numpy()
        
        analysis = {
            'current_state': {cls: prob for cls, prob in zip(self.class_names, current_probs)},
            'progression': {cls: prob for cls, prob in zip(self.class_names, progress_probs)},
            'trends': []
        }
        
        for cls, prog in zip(self.class_names, progress_probs):
            if prog > 0.6:
                analysis['trends'].append(f"{cls}: Worsening")
            elif prog < 0.4:
                analysis['trends'].append(f"{cls}: Improving")
            else:
                analysis['trends'].append(f"{cls}: Stable")
        
        return analysis

# Main execution
if __name__ == "__main__":
    # Initialize complete system
    system = CompleteLungAISystem()
    
    # Generate clinical data first
    clinical_gen = ClinicalDataGenerator()
    train_df = pd.read_csv("../data/splits/train.csv")
    patient_ids = train_df['image_id'].tolist()
    clinical_gen.generate_for_patients(patient_ids, "../data/splits/train.csv")
    
    print("✅ Multi-modal system ready!")
    print("Next: Run training with clinical data fusion")