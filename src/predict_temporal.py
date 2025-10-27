# src/predict_temporal.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt

from temporal_model import TemporalLSTMModel
from temporal_data_loader import get_temporal_transforms

class TemporalPredictor:
    def __init__(self, model_path, sequence_length=2, device='cpu'):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.classes = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
        
        # Load model
        self.model = TemporalLSTMModel(
            sequence_length=sequence_length,
            n_classes=len(self.classes)
        )
        
        # Load model weights with error handling
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✅ Loaded temporal model from {model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("🔄 Using randomly initialized model")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("🔄 Using randomly initialized model")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_temporal_transforms(is_train=False)
    
    def predict_sequence(self, image_paths):
        """Predict for a sequence of images"""
        if len(image_paths) != self.sequence_length:
            raise ValueError(f"Need exactly {self.sequence_length} images, got {len(image_paths)}")
        
        # Load and transform images
        images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
            else:
                print(f"⚠️  Image not found: {img_path}")
                # Create a blank image as fallback
                img_tensor = torch.zeros(3, 224, 224)
                images.append(img_tensor)
        
        # Create sequence: [1, sequence_length, 3, H, W]
        image_sequence = torch.stack(images).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, progress_outputs = self.model(image_sequence)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            progress_probs = torch.sigmoid(progress_outputs).squeeze().cpu().numpy()
        
        return probs, progress_probs
    
    def visualize_prediction(self, image_paths, predictions, output_path):
        """Create visualization of temporal prediction"""
        fig, axes = plt.subplots(2, self.sequence_length + 1, figsize=(15, 8))
        
        # Display input images
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[0, i].imshow(img, cmap='gray')
            else:
                axes[0, i].text(0.5, 0.5, 'Image\nNot Found', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'Scan {i+1}')
            axes[0, i].axis('off')
        
        # Display predictions
        axes[1, 0].axis('off')  # Empty cell for alignment
        
        # Current state probabilities
        current_probs = predictions[0]
        axes[1, 1].barh(self.classes, current_probs, color='skyblue')
        axes[1, 1].set_title('Current State Probabilities')
        axes[1, 1].set_xlim(0, 1)
        
        # Progress predictions
        progress_probs = predictions[1]
        colors = ['red' if x > 0.5 else 'green' for x in progress_probs]
        axes[1, 2].barh(self.classes, progress_probs, color=colors)
        axes[1, 2].set_title('Disease Progress Prediction')
        axes[1, 2].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = TemporalPredictor(
        model_path="../models/temporal_lstm_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example prediction
    image_paths = [
        "../data/processed/00000001_001.png",
        "../data/processed/00000001_002.png"
    ]
    
    # Check if files exist, use any available images if not
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
    
    # If we don't have the specific images, use any two images
    if len(valid_paths) < 2:
        available_images = [f for f in os.listdir("../data/processed") if f.endswith('.png')][:2]
        valid_paths = [f"../data/processed/{img}" for img in available_images]
        print(f"Using available images: {[os.path.basename(p) for p in valid_paths]}")
    
    if len(valid_paths) == 2:
        probs, progress = predictor.predict_sequence(valid_paths)
        
        print("\n🕒 TEMPORAL ANALYSIS RESULTS:")
        print("\nCurrent State Probabilities:")
        for cls, prob in zip(predictor.classes, probs):
            print(f"   {cls:<15}: {prob:.3f}")
        
        print("\nDisease Progress Prediction:")
        for cls, prog in zip(predictor.classes, progress):
            trend = "📈 Worsening" if prog > 0.6 else "📉 Improving" if prog < 0.4 else "➡️ Stable"
            print(f"   {cls:<15}: {prog:.3f} ({trend})")
        
        # Save visualization
        output_path = "../outputs/temporal_prediction.png"
        predictor.visualize_prediction(valid_paths, (probs, progress), output_path)
        print(f"\n✅ Visualization saved to: {output_path}")
    else:
        print("❌ Not enough images found for temporal analysis")