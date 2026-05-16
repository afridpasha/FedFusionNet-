# 🏥 FedFusionNet++: Two-Stage Oral Cancer Detection Models

<div align="center">

![NeuroPlex AI](https://raw.githubusercontent.com/afridpasha/NeuroPlex-AI/main/Logos/Final%20APP%20Logo.png)

### **State-of-the-Art AI Models for Oral Cancer Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-afridpasha-181717?logo=github)](https://github.com/afridpasha/NeuroPlex-AI)

**🔬 Advanced Two-Stage Architecture for OSCC Detection**

*Combining Vision Transformers, Clinical Data, and Explainable AI*

[📖 Full Documentation](https://github.com/afridpasha/NeuroPlex-AI) • [🚀 Quick Start](#-quick-start) • [💻 Demo](https://github.com/afridpasha/NeuroPlex-AI#-demo-credentials)

</div>

---

## 📋 Model Overview

This repository contains the pre-trained models for **FedFusionNet++**, a production-ready two-stage AI system for detecting and staging **Oral Squamous Cell Carcinoma (OSCC)** from histopathology images and clinical data.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **🔬 Two-Stage Architecture** | CNN image analysis + Tabular clinical data prediction |
| **🎨 Macenko Normalization** | Standardizes H&E staining across different labs |
| **🧠 Hybrid Fusion** | Weighted ensemble (0.95×CNN + 0.5×Tabular) |
| **🔍 Explainable AI** | Grad-CAM++ and Layer-CAM visualizations |
| **📊 Uncertainty Quantification** | MC-Dropout with 50 stochastic passes |
| **📈 Survival Prediction** | Kaplan-Meier curves with personalized rates |
| **🎯 High Accuracy** | 96.8% accuracy with hybrid fusion |

---

## 📦 Model Files

### 1️⃣ Stage-1: CNN Model (HetFusionNet v2)

**File:** `hetfusionnet_v2_FINAL.pth` (305.75 MB)

| Specification | Details |
|---------------|---------|
| **Architecture** | Swin-ViT-Small + CrossViT-15 with SE Fusion |
| **Parameters** | ~76 Million |
| **Tensors** | 1,664 tensors |
| **Input Shape** | [1, 3, 240, 240] (RGB image) |
| **Output Shape** | [1, 2] (Normal, OSCC) |
| **Framework** | PyTorch 2.2.0 |
| **Accuracy** | 95.2% |
| **AUC-ROC** | 0.982 |

**Architecture Components:**
- **Swin-ViT-Small**: Hierarchical vision transformer (768 dimensions, 789 tensors)
- **CrossViT-15**: Dual-branch cross-attention (768 dimensions, 857 tensors)
- **SE Fusion Block**: Squeeze-and-excitation with GELU (7 tensors)
- **Classification Head**: 1536→256→2 (11 tensors)
- **MC-Dropout**: 50 forward passes for uncertainty quantification

**Outputs:**
- `diagnosis`: 0 (Normal) or 1 (OSCC)
- `probability`: 0.0 - 1.0
- `confidence`: 0 (LOW) / 1 (MODERATE) / 2 (HIGH)
- `uncertainty`: 0.0 - 0.35

---

### 2️⃣ Stage-2: Tabular Model (RealTabPFN-2.5)

**File:** `stage2_tabular_model.pkl` (~500 KB)

| Specification | Details |
|---------------|---------|
| **Algorithm** | RealTabPFN-2.5 (XGBoost Ensemble) |
| **Input Features** | 20 (16 patient + 4 CNN outputs) |
| **Output Classes** | 5 (Stage 0-4) |
| **Framework** | scikit-learn + XGBoost |
| **Accuracy** | 87.3% |

**Input Features:**
- **Patient Clinical Features (16)**: Age, Gender, Tobacco Use, Alcohol, HPV, Betel Quid, Sun Exposure, Oral Hygiene, Diet, Family History, Immune System, Oral Lesions, Bleeding, Swallowing, Patches, Country
- **CNN Output Features (4)**: diagnosis, probability, confidence, uncertainty

**Outputs:**
- `cancer_stage`: 0-4 (TNM classification)
- `stage_confidence`: 0-100%
- `survival_rate_5yr`: 0-100%
- `treatment_type`: Surgery, Radiation, Chemotherapy, Combined
- `cost_usd`: Estimated treatment cost
- `economic_burden_days`: Treatment duration

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch==2.2.0 torchvision==0.17.0
pip install timm==0.9.12 opencv-python==4.9.0.80
pip install numpy==1.26.3 pillow==10.2.0
pip install scikit-learn==1.4.0 xgboost==2.0.3
```

### Download Models

```python
from huggingface_hub import hf_hub_download

# Download CNN model
cnn_model_path = hf_hub_download(
    repo_id="afridpasha/FedFusionNet-Plus-Plus",
    filename="hetfusionnet_v2_FINAL.pth"
)

# Download Tabular model
tabular_model_path = hf_hub_download(
    repo_id="afridpasha/FedFusionNet-Plus-Plus",
    filename="stage2_tabular_model.pkl"
)
```

### Load Models

```python
import torch
import pickle

# Load CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = torch.load(cnn_model_path, map_location=device)
cnn_model.eval()

# Load Tabular model
with open(tabular_model_path, 'rb') as f:
    tabular_model = pickle.load(f)
```

### Inference Example

```python
from PIL import Image
import torchvision.transforms as transforms

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('path/to/histopathology_image.jpg')
input_tensor = transform(image).unsqueeze(0).to(device)

# Stage-1: CNN Inference
with torch.no_grad():
    cnn_output = cnn_model(input_tensor)
    probabilities = torch.softmax(cnn_output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()

print(f"CNN Prediction: {'OSCC' if prediction == 1 else 'Normal'}")
print(f"Confidence: {confidence:.2%}")

# Stage-2: Tabular Inference
patient_features = [
    55,  # Age
    1,   # Gender (Male=1, Female=0)
    1,   # Tobacco Use (Yes=1, No=0)
    1,   # Alcohol (Yes=1, No=0)
    # ... (16 patient features + 4 CNN outputs)
]

tabular_output = tabular_model.predict([patient_features])
cancer_stage = tabular_output[0]

print(f"Cancer Stage: {cancer_stage}")
```

---

## 📊 Model Performance

### Stage-1 CNN Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.2% |
| **Precision** | 94.8% |
| **Recall** | 95.6% |
| **F1-Score** | 95.2% |
| **AUC-ROC** | 0.982 |
| **Specificity** | 94.7% |
| **Sensitivity** | 95.6% |

### Stage-2 Tabular Model

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.3% |
| **Macro F1** | 86.8% |
| **Weighted F1** | 87.1% |

### Hybrid Fusion System

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.8% |
| **Precision** | 96.5% |
| **Recall** | 97.1% |
| **F1-Score** | 96.8% |
| **AUC-ROC** | 0.991 |

**Improvement:** +1.6% over CNN alone, +9.5% over Tabular alone

---

## 🔬 Technical Details

### Preprocessing Pipeline

1. **Quality Control**
   - Blur detection (Laplacian variance ≥ 100)
   - Tissue content check (≥ 10% tissue)

2. **Macenko Stain Normalization**
   - RGB → Optical Density conversion
   - SVD decomposition for H&E separation
   - Standardization to reference stain matrix

3. **Image Normalization**
   - Resize to 240×240 pixels
   - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Hybrid Fusion Algorithm

```python
# Fusion weights
CNN_WEIGHT = 0.95      # Primary model (image-based)
TABULAR_WEIGHT = 0.5   # Supporting model (clinical data)

# Score calculation
cnn_score = cnn_diagnosis × cnn_probability
tabular_score = (1 if stage > 0 else 0) × stage_confidence

# Weighted fusion
hybrid_score = (CNN_WEIGHT × cnn_score) + (TABULAR_WEIGHT × tabular_score)

# Final decision
final_prediction = 'OSCC' if hybrid_score > 0.5 else 'Normal'

# Confidence calculation
final_confidence = ((CNN_WEIGHT × cnn_probability) + 
                   (TABULAR_WEIGHT × stage_confidence)) / 
                   (CNN_WEIGHT + TABULAR_WEIGHT)
```

---

## 📖 Usage in Full Application

These models are part of the complete **FedFusionNet++** application which includes:

- 🌐 **Web Application**: Flask-based interface with authentication
- 🔍 **Explainable AI**: Grad-CAM++ and Layer-CAM visualizations
- 🗺️ **WSI Analysis**: Tile-based whole slide imaging with heatmaps
- 📊 **SHAP Analysis**: Risk factor contribution analysis
- 📈 **Survival Prediction**: Kaplan-Meier curves with personalized rates
- 💬 **VLM ChatBot**: AI-powered clinical Q&A in 100+ languages
- 📄 **PDF Reports**: Professional clinical report generation

**Full Application:** [GitHub Repository](https://github.com/afridpasha/NeuroPlex-AI)

---

## 🎓 Citation

If you use these models in your research, please cite:

```bibtex
@software{fedfusionnet_plusplus_2024,
  title = {FedFusionNet++: Two-Stage Oral Cancer Detection System},
  author = {Afrid Pasha},
  year = {2024},
  url = {https://github.com/afridpasha/NeuroPlex-AI},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/afridpasha/FedFusionNet-Plus-Plus}}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Afrid Pasha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ Medical Disclaimer

**IMPORTANT:** These models are research tools designed to assist medical professionals. They are **NOT** substitutes for professional medical advice, diagnosis, or treatment.

**Key Points:**
1. **Not FDA Approved**: Not approved for clinical use
2. **Research Use Only**: For research and educational purposes
3. **Professional Oversight Required**: All predictions must be validated by qualified pathologists
4. **No Warranty**: Provided "as is" without warranty
5. **Consult Healthcare Providers**: Always seek professional medical advice

---

## 🙏 Acknowledgments

### Research Papers

1. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
2. **CrossViT**: Chen et al., "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification", ICCV 2021
3. **Grad-CAM++**: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations", WACV 2018
4. **Macenko Normalization**: Macenko et al., "A method for normalizing histology slides for quantitative analysis", ISBI 2009

### Frameworks

- **PyTorch**: Facebook AI Research
- **Timm**: Ross Wightman
- **OpenCV**: Intel Corporation
- **scikit-learn**: scikit-learn developers
- **XGBoost**: DMLC

---

## 👨‍💻 Author

<div align="center">

**Afrid Pasha**

*AI/ML Engineer | Flutter Developer | Healthcare Innovation Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/afridpasha)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/afridpasha)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/afridpasha)

</div>

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/afridpasha/NeuroPlex-AI/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/afridpasha/NeuroPlex-AI/discussions)
- **Email**: afridpasha@example.com

---

<div align="center">

## 🌟 Built with ❤️ for Healthcare Innovation 🌟

**NeuroPlex AI - Transforming Medical Diagnostics with AI**

*"Privacy-preserving AI for a healthier tomorrow"*

---

**Made with 💙 by Afrid Pasha | © 2024 NeuroPlex AI**

[⬆ Back to Top](#-fedfusionnet-two-stage-oral-cancer-detection-models)

</div>
