# src/train_temporal.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from temporal_data_loader import TemporalCXRDataset, get_temporal_transforms
from temporal_model import TemporalLSTMModel

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 2  # Use 2-image sequences (before/after)
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
CLASSES = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']

def calculate_metrics(y_true, y_pred):
    """Calculate multiple metrics"""
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

def train_temporal_model():
    print("🚀 Starting Temporal Model Training...")
    print(f"Device: {DEVICE}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    
    # Data transforms
    train_transform = get_temporal_transforms(is_train=True)
    val_transform = get_temporal_transforms(is_train=False)
    
    # Datasets
    train_ds = TemporalCXRDataset(
        csv_file="../data/splits/train.csv",
        img_dir="../data/processed",
        sequence_length=SEQUENCE_LENGTH,
        transform=train_transform,
        max_patients=500  # Limit for stability
    )
    
    val_ds = TemporalCXRDataset(
        csv_file="../data/splits/val.csv", 
        img_dir="../data/processed",
        sequence_length=SEQUENCE_LENGTH,
        transform=val_transform,
        max_patients=200
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training sequences: {len(train_ds)}")
    print(f"Validation sequences: {len(val_ds)}")
    
    # Model
    model = TemporalLSTMModel(
        sequence_length=SEQUENCE_LENGTH,
        n_classes=len(CLASSES),
        hidden_size=512
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # FIXED: Remove verbose parameter for compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_auc = 0
    best_model_path = "../models/temporal_lstm_model.pth"
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (images, labels, sequence_ids) in enumerate(tqdm(train_loader, desc="Training")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, progress_outputs = model(images)
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
        
        with torch.no_grad():
            for images, labels, sequence_ids in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs, progress_outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_metrics = calculate_metrics(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_metrics['auc_macro'])
        history['val_auc'].append(val_metrics['auc_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}, AUC: {train_metrics['auc_macro']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, AUC: {val_metrics['auc_macro']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['auc_macro'] > best_auc:
            best_auc = val_metrics['auc_macro']
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! AUC: {best_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if epoch > 5 and avg_val_loss > max(history['val_loss'][-5:]):
            print("🛑 Early stopping triggered")
            break
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/temporal_training_history.png')
    plt.close()
    
    print(f"\n🎯 Training completed! Best validation AUC: {best_auc:.4f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    train_temporal_model()