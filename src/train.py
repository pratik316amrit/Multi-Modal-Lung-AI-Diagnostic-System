import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ===============================
# Configurations
# ===============================
DATA_DIR = "../data/processed"   # change if images are in a different folder
SPLIT_DIR = "../data/splits"
CLASS_LIST = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Custom Dataset
# ===============================
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, class_list=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_list = class_list

        # detect if preprocessed CSV format
        if set(class_list).issubset(self.data.columns):
            self.mode = "split"
            # numeric labels
            labels = []
            for c in class_list:
                col = pd.to_numeric(self.data[c], errors="coerce").fillna(0).astype(int)
                labels.append(col.values)
            self.labels = np.vstack(labels).T
            # find the image id column
            if "image_id" in self.data.columns:
                self.img_col = "image_id"
            elif "path" in self.data.columns:
                self.img_col = "path"
            else:
                raise ValueError("No image_id or path column found in CSV!")
        else:
            # fallback for NIH-style CSV
            self.mode = "nih"
            self.data["Finding Labels"] = self.data["Finding Labels"].astype(str)
            self.data["labels_list"] = self.data["Finding Labels"].apply(lambda x: x.split("|"))
            mlb = MultiLabelBinarizer(classes=class_list)
            self.labels = mlb.fit_transform(self.data["labels_list"])
            self.img_col = "Image Index"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row[self.img_col]
        img_path = os.path.join(self.img_dir, os.path.basename(img_name))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()
        return image, label


# ===============================
# Transforms
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================
# Datasets & Loaders
# ===============================
train_ds = ChestXrayDataset(os.path.join(SPLIT_DIR, "train.csv"), DATA_DIR, transform, class_list=CLASS_LIST)
val_ds   = ChestXrayDataset(os.path.join(SPLIT_DIR, "val.csv"), DATA_DIR, transform, class_list=CLASS_LIST)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# Model setup
# ===============================
model = models.resnet50(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, len(CLASS_LIST)),
    nn.Sigmoid()
)
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================
# Training loop
# ===============================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    print(f"\n🌿 Epoch {epoch + 1}/{EPOCHS} ----------------------")
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

    # ===============================
    # Validation
    # ===============================
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            all_labels.append(labels.cpu())
            all_preds.append(outputs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    try:
        auc = roc_auc_score(y_true, y_pred, average="macro")
        print(f"Validation AUC: {auc:.4f}")
    except Exception as e:
        print("AUC computation failed:", e)

# ===============================
# Save model
# ===============================
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/resnet50_lung.pth")
print("\n✅ Model saved to '../models/resnet50_lung.pth'")
