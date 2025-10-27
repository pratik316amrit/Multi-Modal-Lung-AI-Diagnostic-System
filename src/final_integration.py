# src/final_integration.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys
import os

# Add the current directory to path to import your Grad-CAM
sys.path.append('.')

class CompleteAIWithExplanations:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Pneumonia', 'Fibrosis', 'Consolidation', 'No Finding']
        
    def load_gradcam_model(self):
        """Load the model for Grad-CAM (similar to your gradcam.py)"""
        from torchvision import models
        import torch.nn as nn
        
        # Load model (same as your gradcam.py)
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 4),
            nn.Sigmoid()
        )
        
        try:
            state_dict = torch.load("../models/resnet50_lung.pth", map_location="cpu")
            # Handle mismatched keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("fc.0."):
                    new_state_dict[k.replace("fc.0.", "fc.")] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading Grad-CAM model: {e}")
            return None
    
    def generate_gradcam(self, model, image_path, target_class=None):
        """Generate Grad-CAM heatmap (adapted from your gradcam.py)"""
        from torchvision import transforms
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        
        # Grad-CAM implementation
        model.eval()
        
        # Hook for gradients and activations
        gradients = None
        activations = None
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        # Register hooks to the target layer (layer4 for ResNet50)
        target_layer = model.layer4[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = model(input_tensor)
        
        # If no target class specified, use the highest probability class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients and backward pass for target class
        model.zero_grad()
        target = output[0][target_class]
        target.backward()
        
        # Generate CAM
        if gradients is not None and activations is not None:
            weights = torch.mean(gradients, dim=(2, 3))[0]  # Global average pooling of gradients
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
            
            for i, w in enumerate(weights):
                cam += w * activations[0, i, :, :]
            
            cam = torch.relu(cam)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            cam = cam.numpy()
            
            # Resize to match original image
            cam = cv2.resize(cam, (224, 224))
        else:
            # Fallback: create a blank CAM
            cam = np.zeros((224, 224))
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return cam, np.array(img.resize((224, 224))) / 255.0
    
    def create_gradcam_overlay(self, cam, original_img, alpha=0.5):
        """Create Grad-CAM overlay on original image"""
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Ensure both images have same dimensions
        if original_img.shape[:2] != heatmap.shape[:2]:
            original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
        
        # Blend images
        overlay = heatmap * alpha + original_img * (1 - alpha)
        overlay = overlay / np.max(overlay)  # Normalize
        
        return overlay
    
    def create_complete_visualization(self, image_path, clinical_data, predictions, attention_weights):
        """Create complete visualization with integrated Grad-CAM"""
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        image_array = np.array(original_image)
        
        # Generate Grad-CAM
        gradcam_model = self.load_gradcam_model()
        if gradcam_model is not None:
            # Use the class with highest probability for Grad-CAM
            target_class = max(predictions.items(), key=lambda x: x[1])[0]
            target_class_idx = self.class_names.index(target_class)
            
            cam, processed_img = self.generate_gradcam(gradcam_model, image_path, target_class_idx)
            gradcam_overlay = self.create_gradcam_overlay(cam, processed_img)
        else:
            # Fallback if Grad-CAM fails
            gradcam_overlay = np.zeros((224, 224, 3))
            print("⚠️  Grad-CAM not available, using placeholder")
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Original image
        plt.subplot(2, 4, 1)
        plt.imshow(image_array, cmap='gray')
        plt.title('Original Chest X-ray', fontweight='bold', fontsize=12)
        plt.axis('off')
        
        # 2. Grad-CAM heatmap
        plt.subplot(2, 4, 2)
        if gradcam_model is not None:
            plt.imshow(gradcam_overlay)
            plt.title(f'Grad-CAM: {target_class}', fontweight='bold', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=12, fontweight='bold')
            plt.title('AI Attention Map', fontweight='bold', fontsize=12)
        plt.axis('off')
        
        # 3. Disease probabilities
        plt.subplot(2, 4, 3)
        diseases = list(predictions.keys())
        probs = list(predictions.values())
        colors = ['#ff4444' if p > 0.7 else '#ffaa00' if p > 0.3 else '#44ff44' for p in probs]
        bars = plt.barh(diseases, probs, color=colors, alpha=0.8)
        plt.xlim(0, 1)
        plt.title('Disease Probabilities', fontweight='bold', fontsize=12)
        plt.xlabel('Probability')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            color = 'white' if prob > 0.5 else 'black'
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10, fontweight='bold',
                    color=color)
        
        # 4. Multi-modal attention weights
        plt.subplot(2, 4, 4)
        modalities = ['X-ray Image', 'Clinical Data']
        colors = ['#4285F4', '#EA4335']  # Google blue and red
        wedges, texts, autotexts = plt.pie(attention_weights, labels=modalities, 
                                          autopct='%1.1f%%', colors=colors,
                                          startangle=90)
        plt.title('Multi-Modal Feature Importance', fontweight='bold', fontsize=12)
        
        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 5. Clinical data visualization
        plt.subplot(2, 4, 5)
        if clinical_data:
            factors = ['Age', 'Smoking', 'Oxygen', 'Fever', 'Cough', 'Breath', 'Pain']
            values = [
                clinical_data.get('age', 0)/100,
                clinical_data.get('smoking_history', 0)/2,
                clinical_data.get('oxygen_saturation', 0)/100,
                clinical_data.get('fever', 0),
                clinical_data.get('cough', 0),
                clinical_data.get('shortness_of_breath', 0),
                clinical_data.get('chest_pain', 0)
            ]
            
            # Color code based on risk
            colors = []
            for i, (factor, value) in enumerate(zip(factors, values)):
                if factor == 'Oxygen' and value < 0.92:  # Oxygen < 92%
                    colors.append('#ff4444')
                elif factor == 'Smoking' and value > 0.5:  # Current smoker
                    colors.append('#ffaa00')
                elif factor in ['Fever', 'Cough', 'Breath', 'Pain'] and value > 0:
                    colors.append('#ffaa00')
                else:
                    colors.append('#4285F4')
            
            bars = plt.bar(factors, values, color=colors, alpha=0.8)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.title('Clinical Factors', fontweight='bold', fontsize=12)
            plt.ylabel('Normalized Value')
            
            # Add value annotations
            for bar, value in zip(bars, values):
                if value > 0.1: # Only show text for significant values
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Risk assessment
        plt.subplot(2, 4, 6)
        high_risk = sum(1 for p in probs if p > 0.7)
        medium_risk = sum(1 for p in probs if 0.3 < p <= 0.7)
        low_risk = sum(1 for p in probs if p <= 0.3)
        
        risk_data = [low_risk, medium_risk, high_risk]
        risk_labels = ['Low', 'Medium', 'High']
        risk_colors = ['#44ff44', '#ffaa00', '#ff4444']
        
        wedges, texts, autotexts = plt.pie(risk_data, labels=risk_labels, colors=risk_colors, 
                                          autopct='%1.0f%%', startangle=90)
        plt.title('Overall Risk Assessment', fontweight='bold', fontsize=12)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 7. Confidence metrics
        plt.subplot(2, 4, 7)
        metrics = ['Model Confidence', 'Data Quality', 'Clinical Correlation']
        
        # Calculate actual confidence scores
        max_prob = max(probs)
        data_quality = 0.8  # Could be calculated based on image quality
        clinical_corr = 0.7 + (attention_weights[1] * 0.3)  # Higher if clinical data is important
        
        scores = [max_prob, data_quality, clinical_corr]
        colors = ['#4285F4', '#34A853', '#FBBC05']  # Google colors
        
        bars = plt.barh(metrics, scores, color=colors, alpha=0.8)
        plt.xlim(0, 1)
        plt.title('System Confidence', fontweight='bold', fontsize=12)
        plt.xlabel('Score')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=10, fontweight='bold')
        
        # 8. Recommendations
        plt.subplot(2, 4, 8)
        recommendations = []
        
        # Medical recommendations based on predictions and clinical data
        if predictions.get('Pneumonia', 0) > 0.5:
            recommendations.append('• Consider antibiotic therapy')
            if clinical_data.get('fever', 0):
                recommendations.append('• Monitor temperature closely')
        
        if predictions.get('Fibrosis', 0) > 0.5:
            recommendations.append('• Pulmonary function tests')
            recommendations.append('• CT scan for confirmation')
        
        if predictions.get('Consolidation', 0) > 0.5:
            recommendations.append('• Further imaging recommended')
        
        if clinical_data and clinical_data.get('oxygen_saturation', 0) < 92:
            recommendations.append('• Oxygen supplementation')
            recommendations.append('• Monitor respiratory rate')
        
        if clinical_data.get('smoking_history', 0) == 2:  # Current smoker
            recommendations.append('• Smoking cessation counseling')
        
        # If no significant findings
        if not any(p > 0.3 for p in predictions.values()):
            recommendations.append('• No significant abnormalities detected')
            recommendations.append('• Routine follow-up recommended')
        
        # Ensure we have at least some recommendations
        if not recommendations:
            recommendations.append('• Clinical correlation recommended')
            recommendations.append('• Follow-up as needed')
        
        plt.axis('off')
        plt.title('Clinical Recommendations', fontweight='bold', fontsize=12)
        
        # Add recommendations text with proper formatting
        recommendation_text = '\n'.join(recommendations)
        plt.text(0.05, 0.95, 'Recommendations:', fontweight='bold', fontsize=11,
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.85, recommendation_text, fontsize=10,
                transform=plt.gca().transAxes, verticalalignment='top',
                linespacing=1.5)
        
        # Main title
        primary_diagnosis = max(predictions.items(), key=lambda x: x[1])
        plt.suptitle(f'COMPREHENSIVE LUNG AI DIAGNOSTIC REPORT\n'
                     f'Primary Finding: {primary_diagnosis[0]} ({primary_diagnosis[1]:.1%} confidence)', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save figure
        output_path = f"../outputs/comprehensive_report_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Comprehensive report with Grad-CAM saved: {output_path}")
        return output_path

def main():
    print("🎯 GENERATING COMPREHENSIVE DIAGNOSTIC REPORTS WITH GRAD-CAM...")
    
    # Example usage
    ai_system = CompleteAIWithExplanations()
    
    # Example data (replace with actual model predictions)
    example_predictions = {
        'Pneumonia': 0.76,
        'Fibrosis': 0.34,
        'Consolidation': 0.23,
        'No Finding': 0.12
    }
    
    example_clinical = {
        'age': 67,
        'smoking_history': 2,
        'oxygen_saturation': 88.5,
        'fever': 1,
        'cough': 1,
        'shortness_of_breath': 1,
        'chest_pain': 0
    }
    
    example_attention = [0.85, 0.15]  # [image, clinical]
    
    # Get a sample image
    sample_images = list(Path("../data/processed").glob("*.png"))[:1]
    if sample_images:
        image_path = str(sample_images[0])
        print(f"📊 Generating report for: {Path(image_path).name}")
        
        report_path = ai_system.create_complete_visualization(
            image_path, example_clinical, example_predictions, example_attention
        )
        print(f"📊 Sample report generated: {report_path}")
    else:
        print("❌ No sample images found in ../data/processed/")

if __name__ == "__main__":
    main()