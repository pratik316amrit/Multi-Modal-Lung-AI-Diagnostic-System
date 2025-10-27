import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from datasets.temporal_dataset import TemporalLungDataset
from models.temporal_resnet import TemporalResNet as TemporalLungModel

from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_CSV = "../data/raw/chestxray14/Data_Entry_subset.csv"

IMG_DIR = "../data/processed/"
MODEL_PATH = "../models/temporal_resnet_lung.pth"
SEQ_LEN = 3
BATCH_SIZE = 8

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = TemporalLungDataset(DATA_CSV, IMG_DIR, transform=transform, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TemporalLungModel(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print("\n🧾 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Pneumonia", "Fibrosis", "Consolidation", "No Finding"]))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
