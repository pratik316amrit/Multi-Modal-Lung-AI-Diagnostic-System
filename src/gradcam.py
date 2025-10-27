#src/gradcam.py
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image

# ====== PATHS ======
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet50_lung.pth")
IMG_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "gradcam")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== GRADCAM CLASS ======
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

# ====== LOAD MODEL ======
# ====== LOAD MODEL ======
model = models.resnet50(weights=None)

# Define same FC head as training
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 4),
    torch.nn.Sigmoid()
)

state_dict = torch.load(MODEL_PATH, map_location="cpu")

# Handle mismatched keys gracefully
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("fc.0."):
        new_state_dict[k.replace("fc.0.", "fc.")] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()


target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# ====== PICK AN IMAGE ======
image_name = "00000005_000.png"   # change this as you wish
image_path = os.path.join(IMG_DIR, image_name)

img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# ====== GENERATE GRADCAM ======
cam = gradcam.generate(input_tensor)

# ====== OVERLAY HEATMAP ======
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
input_img = np.array(img.resize((224, 224))) / 255.0
overlay = heatmap * 0.5 + input_img * 0.5

# ====== SAVE RESULT ======
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(overlay)
plt.axis('off')

out_path = os.path.join(OUTPUT_DIR, f"gradcam_{image_name}")
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"✅ Grad-CAM saved to: {out_path}")
