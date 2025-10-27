#src/predict.py
import torch, cv2, os, numpy as np, argparse
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from simple_lung_mask import make_lung_mask
from overlay_gradcam_with_mask import GradCAM

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "resnet50_lung.pth"
OUT_DIR = BASE_DIR / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Labels (same order as training) ---
CLASSES = ["Pneumonia", "Fibrosis", "Consolidation", "No Finding"]

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Load model ---
model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, len(CLASSES)), torch.nn.Sigmoid())
state = torch.load(MODEL_PATH, map_location="cpu")
state = {k.replace("fc.0.", "fc."): v for k,v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()

gradcam = GradCAM(model, model.layer4[-1])

def predict_and_visualize(image_path):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(inp)[0].numpy()
    print("\n🩺 Prediction Results:")
    for cls, p in zip(CLASSES, preds):
        print(f"   {cls:<15}: {p:.3f}")
    cam = gradcam(inp)
    mask = make_lung_mask(np.array(img))
    hm = cv2.applyColorMap(np.uint8(255*(cam*mask)), cv2.COLORMAP_JET)/255.0
    img_np = np.array(img.resize((224,224)))/255.0
    blended = cv2.addWeighted(img_np, 0.6, hm, 0.4, 0)
    out_path = OUT_DIR / f"gradcam_{Path(image_path).stem}.png"
    plt.imsave(out_path, blended)
    print(f"\n✅ Grad-CAM saved at: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    args = parser.parse_args()
    predict_and_visualize(args.image)
