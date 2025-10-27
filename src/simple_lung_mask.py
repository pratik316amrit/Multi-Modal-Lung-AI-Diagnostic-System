import os, cv2, numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs" / "masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_lung_mask(img_np, display_size=(224, 224)):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    cl = clahe.apply(gray)

    # lungs are darker → invert after threshold
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 2)
    open_ = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, 1)

    contours, _ = cv2.findContours(open_.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(open_)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = (mask > 127).astype(np.float32)
    mask = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST)
    return mask

def process_and_save(image_name):
    img_path = IMG_DIR / image_name
    img = Image.open(img_path).convert("RGB")
    mask = make_lung_mask(np.array(img))
    out_path = OUT_DIR / f"mask_{image_name}"
    cv2.imwrite(str(out_path), (mask * 255).astype("uint8"))
    return out_path

if __name__ == "__main__":
    files = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg")])[:10]
    for f in files:
        print("Processing", f)
        print("→", process_and_save(f))
