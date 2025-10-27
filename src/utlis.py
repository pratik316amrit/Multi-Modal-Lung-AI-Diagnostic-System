# src/utils.py
import torch
import numpy as np
import cv2
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.5):
    """
    original_img: numpy uint8 HxWx3 (0-255)
    heatmap: float HxW in 0-1
    """
    hm = cv2.applyColorMap((heatmap*255).astype('uint8'), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(hm, alpha, original_img, 1-alpha, 0)
    return overlay

def save_img(img_arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_arr[:,:,::-1])  # RGB to BGR for cv2

def compute_metrics(y_true, y_proba, threshold=0.5):
    from sklearn.metrics import roc_auc_score, accuracy_score
    # y_proba shape [N, C], y_true [N, C]
    aucs = []
    for c in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:,c], y_proba[:,c])
        except:
            auc = float('nan')
        aucs.append(auc)
    preds = (y_proba >= threshold).astype(int)
    acc = (preds == y_true).mean()
    return {'aucs': aucs, 'accuracy': acc}
