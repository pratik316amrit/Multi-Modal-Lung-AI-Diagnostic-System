# src/model.py
import torch
import torch.nn as nn
import timm

class CXRClassifier(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b0', pretrained=True, n_classes=4, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_feats = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        feat = self.backbone(x)  # shape [B, in_feats]
        out = self.head(feat)
        return out, feat  # return logits and feature vector
