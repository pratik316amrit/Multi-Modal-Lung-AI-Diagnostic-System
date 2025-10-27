#src/overlay_gradcam_with_mask.py
import os, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from simple_lung_mask import make_lung_mask

BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "data" / "processed"
MASK_DIR = BASE_DIR / "outputs" / "masks"
OUT_DIR = BASE_DIR / "outputs" / "gradcam_masked"
MODEL_PATH = BASE_DIR / "models" / "resnet50_lung.pth"
SPLIT_CSV = BASE_DIR / "data" / "splits" / "val.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class GradCAM:
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.gradients = self.activations = None
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)
    def _forward_hook(self, m, i, o): self.activations = o.detach()
    def _backward_hook(self, m, gi, go): self.gradients = go[0].detach()
    def __call__(self, x, target=None):
        out = self.model(x)
        if target is None: target = out.argmax(dim=1).item()
        self.model.zero_grad(); out[0, target].backward()
        w = self.gradients[0].mean(dim=(1,2)).cpu().numpy()
        a = self.activations[0].cpu().numpy()
        cam = np.maximum(np.tensordot(w, a, axes=1), 0)
        cam = cv2.resize(cam, (224,224)); cam = (cam - cam.min())/(cam.max()+1e-8)
        return cam

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 4), torch.nn.Sigmoid())
state = torch.load(MODEL_PATH, map_location="cpu")
state = {k.replace("fc.0.", "fc."): v for k,v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()

gradcam = GradCAM(model, model.layer4[-1])

def overlay(cam, mask, img, alpha=0.5):
    hm = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)/255.0
    hm_mask = cv2.applyColorMap(np.uint8(255*(cam*mask)), cv2.COLORMAP_JET)/255.0
    img = np.array(img.resize((224,224)))/255.0
    return hm_mask*alpha + img*(1-alpha)

df = pd.read_csv(SPLIT_CSV)
for name in df["path"].head(10):
    img_path = IMG_DIR / name
    img = Image.open(img_path).convert("RGB")
    mask_path = MASK_DIR / f"mask_{name}"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)/255.0
    else:
        mask = make_lung_mask(np.array(img))
    inp = transform(img).unsqueeze(0)
    cam = gradcam(inp)
    overlay_img = overlay(cam, mask, img)
    out_path = OUT_DIR / f"gradcam_masked_{name}"
    plt.imsave(out_path, overlay_img)
    print("✅ saved:", out_path)
