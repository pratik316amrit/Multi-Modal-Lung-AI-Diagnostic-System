# src/temporal_model.py
import torch
import torch.nn as nn
import timm

class TemporalLSTMModel(nn.Module):
    def __init__(self, backbone_name='resnet50', sequence_length=2, 
                 n_classes=4, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        
        # CNN backbone for feature extraction
        self.backbone = timm.create_model(backbone_name, pretrained=True, 
                                        num_classes=0, global_pool='avg')
        self.feature_dim = self.backbone.num_features
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        # Progress prediction (change between first and last scan)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)  # Predict change for each class
        )
        
    def forward(self, x, return_features=False):
        # x shape: [batch_size, sequence_length, 3, H, W]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Extract CNN features for each frame
        cnn_features = []
        for t in range(seq_len):
            frame_features = self.backbone(x[:, t])  # [batch_size, feature_dim]
            cnn_features.append(frame_features)
        
        # Stack: [batch_size, seq_len, feature_dim]
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.transpose(1, 2)  # [batch_size, 1, seq_len]
        
        # Apply attention
        context_vector = torch.bmm(attention_weights, lstm_out)  # [batch_size, 1, hidden_size]
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(context_vector)
        
        # Progress prediction (difference between first and last)
        first_last = torch.cat([lstm_out[:, 0], lstm_out[:, -1]], dim=1)
        progress_logits = self.progress_head(first_last)
        
        if return_features:
            return logits, progress_logits, context_vector, attention_weights.squeeze(1)
        else:
            return logits, progress_logits

# Simple 3D CNN alternative
class Simple3DCNN(nn.Module):
    def __init__(self, n_classes=4, sequence_length=2):
        super().__init__()
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, 3, H, W]
        x = x.transpose(1, 2)  # [batch_size, 3, sequence_length, H, W]
        
        features = self.conv3d(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        
        return output