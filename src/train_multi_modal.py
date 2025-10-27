# src/train_multi_modal.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from multi_modal_loader import MultiModalCXRDataset
from multi_modal_model import MultiModalFusionModel
from temporal_data_loader import get_temporal_transforms

def calculate_metrics(y_true, y_pred):
    """Calculate metrics with error handling"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    try:
        # AUC for each class
        auc_scores = []
        for i in range(y_true.shape[1]):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        metrics['auc_macro'] = np.mean(auc_scores)
        metrics['auc_per_class'] = auc_scores
        
        # Accuracy (threshold at 0.5)
        y_pred_binary = (y_pred > 0.5).astype(float)
        accuracy = accuracy_score(y_true, y_pred_binary)
        metrics['accuracy'] = accuracy
        
    except Exception as e:
        print(f"Metrics calculation warning: {e}")
        metrics = {'auc_macro': 0.5, 'accuracy': 0.0}
    
    return metrics

def train_multi_modal_system():
    print("🚀 Starting Multi-Modal Training Pipeline...")
    
    # Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 1e-4
    CLASSES = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
    
    # Create data transforms
    train_transform = get_temporal_transforms(is_train=True)
    val_transform = get_temporal_transforms(is_train=False)
    
    # Create datasets
    print("📁 Loading datasets...")
    train_dataset = MultiModalCXRDataset(
        csv_file="../data/splits/train.csv",
        img_dir="../data/processed",
        clinical_csv="../data/clinical/clinical_data.csv",
        transform=train_transform
    )
    
    val_dataset = MultiModalCXRDataset(
        csv_file="../data/splits/val.csv",
        img_dir="../data/processed", 
        clinical_csv="../data/clinical/clinical_data.csv",
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = MultiModalFusionModel(num_classes=len(CLASSES)).to(DEVICE)
    
    # Loss function with class weighting
    pos_weight = torch.tensor([2.0, 2.0, 2.0, 0.7]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [],
        'image_attention': [], 'clinical_attention': []
    }
    
    best_auc = 0
    best_model_path = "../models/multi_modal_fusion.pth"
    
    print("\n🎯 Starting training...")
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (images, clinical, labels, _) in enumerate(tqdm(train_loader, desc="Training")):
            images, clinical, labels = images.to(DEVICE), clinical.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, attn_weights, _ = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions
            train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.append(labels.detach().cpu().numpy())
        
        # Calculate training metrics
        avg_train_loss = running_loss / len(train_loader)
        train_preds = np.vstack(train_preds)
        train_labels = np.vstack(train_labels)
        train_metrics = calculate_metrics(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        image_attentions, clinical_attentions = [], []
        
        with torch.no_grad():
            for images, clinical, labels, _ in tqdm(val_loader, desc="Validation"):
                images, clinical, labels = images.to(DEVICE), clinical.to(DEVICE), labels.to(DEVICE)
                outputs, attn_weights, _ = model(images, clinical)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                
                # Handle attention weights properly
                attn_np = attn_weights.cpu().numpy()
                if attn_np.ndim == 2:  # [batch_size, 2]
                    image_attentions.extend(attn_np[:, 0])
                    clinical_attentions.extend(attn_np[:, 1])
                else:
                    # Fallback if attention shape is different
                    image_attentions.extend([0.5] * len(images))
                    clinical_attentions.extend([0.5] * len(images))
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_metrics = calculate_metrics(val_labels, val_preds)
        
        # Calculate average attention weights
        avg_image_attention = np.mean(image_attentions) if image_attentions else 0.5
        avg_clinical_attention = np.mean(clinical_attentions) if clinical_attentions else 0.5
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_metrics['auc_macro'])
        history['val_auc'].append(val_metrics['auc_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['image_attention'].append(avg_image_attention)
        history['clinical_attention'].append(avg_clinical_attention)
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}, AUC: {train_metrics['auc_macro']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, AUC: {val_metrics['auc_macro']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Attention - Image: {avg_image_attention:.3f}, Clinical: {avg_clinical_attention:.3f}")
        
        # Save best model
        if val_metrics['auc_macro'] > best_auc:
            best_auc = val_metrics['auc_macro']
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! AUC: {best_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if epoch > 10 and avg_val_loss > max(history['val_loss'][-5:]):
            print("🛑 Early stopping triggered")
            break
    
    # Plot training history
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Multi-Modal Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('Multi-Modal AUC')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Multi-Modal Accuracy')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.plot(history['image_attention'], label='Image Attention')
    plt.plot(history['clinical_attention'], label='Clinical Attention')
    plt.title('Attention Weights')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/multi_modal_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎯 Multi-Modal Training Completed!")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Training plot saved to: ../outputs/multi_modal_training_history.png")
    
    # Final attention analysis
    print(f"\n🔍 Final Attention Analysis:")
    print(f"Average Image Attention: {np.mean(history['image_attention']):.3f}")
    print(f"Average Clinical Attention: {np.mean(history['clinical_attention']):.3f}")
    
    return model, history

if __name__ == "__main__":
    train_multi_modal_system()