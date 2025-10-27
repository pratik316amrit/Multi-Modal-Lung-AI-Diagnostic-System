# src/multi_modal_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes=4, clinical_dim=8, fusion_dim=256, dropout=0.3):
        super().__init__()
        
        # Image branch (CNN) - using ResNet50
        self.image_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.image_backbone = nn.Sequential(*list(self.image_backbone.children())[:-1])
        image_features = 2048  # resnet50
        
        # Clinical data branch
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(image_features + 32, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(image_features + 32, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # 2 modalities: image and clinical
            nn.Softmax(dim=1)
        )
    
    def forward(self, image, clinical):
        batch_size = image.size(0)
        
        # Image features
        image_feat = self.image_backbone(image)
        image_feat = image_feat.view(batch_size, -1)
        
        # Clinical features
        clinical_feat = self.clinical_net(clinical)
        
        # Concatenate features
        combined = torch.cat([image_feat, clinical_feat], dim=1)
        
        # Attention weights - ensure proper shape [batch_size, 2]
        attn_weights = self.attention(combined)
        
        # Apply attention - split into image and clinical components
        image_portion = image_feat.shape[1]
        clinical_portion = clinical_feat.shape[1]
        
        # Separate the combined features
        image_combined = combined[:, :image_portion]
        clinical_combined = combined[:, image_portion:]
        
        # Apply attention weights
        image_attn = attn_weights[:, 0].unsqueeze(1) * image_combined
        clinical_attn = attn_weights[:, 1].unsqueeze(1) * clinical_combined
        
        # Fusion - combine attended features
        fused = torch.cat([image_attn, clinical_attn], dim=1)
        fused_features = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, attn_weights, fused_features

# Simplified version for debugging
class SimpleMultiModalModel(nn.Module):
    def __init__(self, num_classes=4, clinical_dim=8, dropout=0.3):
        super().__init__()
        
        # Image branch
        self.image_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.image_backbone = nn.Sequential(*list(self.image_backbone.children())[:-1])
        image_features = 2048
        
        # Clinical branch
        self.clinical_net = nn.Linear(clinical_dim, 32)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(image_features + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, clinical):
        # Image features
        image_feat = self.image_backbone(image)
        image_feat = image_feat.view(image_feat.size(0), -1)
        
        # Clinical features
        clinical_feat = self.clinical_net(clinical)
        
        # Combine
        combined = torch.cat([image_feat, clinical_feat], dim=1)
        output = self.classifier(combined)
        
        # Return dummy attention weights for compatibility
        batch_size = image.size(0)
        attn_weights = torch.ones(batch_size, 2) * 0.5  # Equal attention
        
        return output, attn_weights, combined