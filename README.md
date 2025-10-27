# 🏥 Multi-Modal Lung AI Diagnostic System


> Advanced AI-powered system for early diagnosis and explainable prognosis of lung disorders using multi-modal data fusion (X-ray + Clinical Data + Temporal Analysis)

## 🌟 Features

### 🎯 Multi-Modal AI Fusion
- **X-ray Image Analysis**: Deep learning-based feature extraction using ResNet-50
- **Clinical Data Integration**: Patient demographics, symptoms, and risk factors
- **Attention Mechanism**: Learnable weights for feature importance between modalities
- **Temporal Analysis**: LSTM-based progression prediction using sequential scans

### 🔍 Explainable AI (XAI)
- **Grad-CAM Heatmaps**: Visual explanations showing where the AI focuses in X-rays
- **Attention Weights**: Quantifiable feature importance between image and clinical data
- **Confidence Scores**: Model certainty metrics for each prediction
- **Clinical Correlation**: AI recommendations with medical reasoning

### 💻 User-Friendly Interface
- **Streamlit Web App**: Interactive dashboard for real-time analysis
- **Batch Processing**: Analyze multiple X-rays simultaneously
- **Comprehensive Reports**: 8-panel diagnostic visualization with export options
- **Real-time Processing**: Instant AI predictions with live visualizations

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Multi-Modal AUC** | 0.8124 | Overall diagnostic accuracy |
| **Image Attention** | 85.1% | Model reliance on X-ray features |
| **Clinical Attention** | 14.9% | Model reliance on clinical data |
| **Inference Speed** | < 2s | Real-time prediction capability |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) with CUDA support
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/pratik316amrit/Multi-Modal-Lung-AI-Diagnostic-System/
cd lung-ai-diagnostic-system
```

2. **Run the StreamLit app**
```bash
streamlit run app.py
```
