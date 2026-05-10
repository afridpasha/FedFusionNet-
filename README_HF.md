---
title: FedFusionNet++ Oral Cancer Detection
emoji: 🏥
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🏥 FedFusionNet++: Advanced Two-Stage Oral Cancer Detection System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-red.svg)](https://pytorch.org/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🔬 State-of-the-Art AI-Powered Oral Squamous Cell Carcinoma (OSCC) Detection Platform**

*Combining Deep Learning, Explainable AI, Vision-Language Models, and Clinical Intelligence*

</div>

---

## 🎯 Project Overview

**FedFusionNet++** is an enterprise-grade, production-ready web application designed for accurate detection and staging of **Oral Squamous Cell Carcinoma (OSCC)** using cutting-edge artificial intelligence. This system represents a breakthrough in medical AI by combining multiple state-of-the-art technologies into a unified diagnostic platform.

### 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **🔬 Two-Stage Architecture** | Combines CNN image analysis with clinical data prediction for superior accuracy |
| **🎨 Macenko Stain Normalization** | Standardizes H&E histopathology images across different labs and scanners |
| **🧠 Hybrid Fusion** | Weighted ensemble (0.95×CNN + 0.5×Tabular) for robust predictions |
| **🔍 Explainable AI (XAI)** | Grad-CAM++ and Layer-CAM visualizations for transparent decision-making |
| **🗺️ WSI Spatial Analysis** | Tile-based whole slide imaging with color-coded cancer probability heatmaps |
| **💬 Vision-Language Model** | AI-powered clinical narratives and conversational Q&A in 100+ languages |
| **📊 SHAP Analysis** | Risk factor contribution analysis with what-if scenarios |
| **📈 Survival Prediction** | Kaplan-Meier curves with personalized 1/3/5/10-year survival rates |
| **🌐 Multi-Language Support** | Clinical reports in 100+ languages using Gemini 2.5 Flash & Groq Llama 3.3 |
| **📱 Fully Responsive** | Mobile, tablet, and desktop optimized interface |

---

## 🚀 Quick Start

### Demo Credentials

For testing purposes, use these demo credentials:

**Hospital Login:**
- **Email:** `demo@hospital.com`
- **Password:** `demo123`

**Or create a new hospital account through the signup page.**

---

## 🔧 Technical Specifications

### Stage-1: CNN Model
- **Architecture:** Dual Vision Transformer (Swin-ViT-Small + CrossViT-15)
- **Model Size:** 305.75 MB
- **Parameters:** ~76 Million
- **Input:** 240×240 RGB histopathology images
- **Output:** OSCC/Normal classification with confidence and uncertainty

### Stage-2: Tabular Model
- **Algorithm:** RealTabPFN-2.5 (XGBoost-based)
- **Input Features:** 20 (16 patient + 4 CNN outputs)
- **Output:** Cancer stage (0-4), survival rate, treatment recommendations

### Hybrid Fusion
```python
hybrid_score = (0.95 × cnn_score) + (0.5 × tabular_score)
final_prediction = 'OSCC' if hybrid_score > 0.5 else 'Normal'
```

---

## 📊 Performance Metrics

- **Accuracy:** 95%+ confidence with uncertainty quantification
- **Processing Time:** < 30 seconds per case (including XAI, WSI, and VLM)
- **Supported Languages:** 100+ languages for clinical reports
- **Deployment:** Production-ready Flask web application with REST API

---

## 🛠️ Technology Stack

### Backend
- **Python 3.10+**
- **Flask 3.0** - Web framework
- **PyTorch 2.2** - Deep learning
- **timm 0.9.12** - Vision transformers
- **OpenCV 4.9** - Image processing
- **scikit-learn 1.4** - ML utilities
- **lifelines 0.27.8** - Survival analysis
- **Google Gemini 2.5 Flash** - VLM (FREE)
- **Groq Llama 3.3 70B** - VLM fallback (FREE)

### Frontend
- **HTML5, CSS3, JavaScript ES6+**
- **Tailwind CSS 3.4** - Utility-first CSS
- **Chart.js** - Interactive visualizations
- **Font Awesome 6.4** - Icons
- **Plotly.js** - Advanced charts

---

## 📦 Installation & Deployment

### Local Development

```bash
# Clone repository
git clone https://github.com/afridpasha/FedFusionNet-.git
cd FedFusionNet-

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export GROQ_API_KEY="your_groq_api_key"

# Run application
python -m flask run --host=0.0.0.0 --port=7860
```

### Docker Deployment

```bash
# Build Docker image
docker build -t fedfusionnet-plus-plus .

# Run container
docker run -p 7860:7860 \
  -e GEMINI_API_KEY="your_key" \
  -e GROQ_API_KEY="your_key" \
  fedfusionnet-plus-plus
```

### Hugging Face Spaces

This application is deployed on Hugging Face Spaces using Docker SDK.

**Required Secrets:**
- `GEMINI_API_KEY` - Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- `GROQ_API_KEY` - Get free API key from [Groq Console](https://console.groq.com/)

---

## 🎮 Usage Guide

1. **Login/Signup** - Create hospital account or use demo credentials
2. **Dashboard** - View prediction history and statistics
3. **Predict** - Upload histopathology image + enter 16 patient features
4. **Results** - View comprehensive analysis:
   - Final diagnosis with confidence
   - XAI heatmaps (Grad-CAM++, Layer-CAM)
   - WSI spatial analysis
   - SHAP risk factor analysis
   - Survival curves (Kaplan-Meier)
   - Clinical narrative (AI-generated)
   - ChatBot Q&A assistant
5. **History** - Search patient records by Patient ID
6. **Analytics** - View hospital-wide statistics and trends

---

## 🔬 Advanced Features

### Explainable AI (XAI)
- **Grad-CAM++:** Advanced gradient-based attention visualization
- **Layer-CAM:** Alternative visualization for cross-validation
- **Processing:** Adaptive thresholding, sharpening, contrast enhancement

### Whole Slide Imaging (WSI)
- **Tile-Based Processing:** 224×224 patches with batch prediction
- **Spatial Heatmap:** Color-coded visualization (Blue=Normal, Red=Cancer)
- **Statistics:** Total tiles, cancer distribution, average confidence

### SHAP Risk Factor Analysis
- **Feature Importance:** Top risk and protective factors
- **What-If Scenarios:** Tobacco cessation, alcohol reduction simulations
- **Actionable Insights:** Personalized risk reduction strategies

### Survival Analysis
- **Kaplan-Meier Curves:** Personalized survival probability over time
- **Milestone Predictions:** 1/3/5/10-year survival rates
- **Population Comparison:** Benchmarking against stage-matched cohorts

### Vision-Language Model (VLM)
- **Clinical Narrative Generation:** Comprehensive medical reports
- **Conversational Q&A:** Context-aware medical assistant
- **Multi-Language Translation:** 100+ languages supported

---

## 📄 API Documentation

### POST /api/predict
Perform cancer detection prediction

**Request:**
```json
{
  "image": "base64_encoded_image",
  "patient_data": {
    "Age": 45,
    "Gender": "Male",
    "Tobacco Use": "Yes",
    ...
  }
}
```

**Response:**
```json
{
  "success": true,
  "prediction_id": "PAT-20240115123045",
  "final_prediction": "OSCC",
  "final_confidence": 92.5,
  "risk_level": "HIGH",
  ...
}
```

### GET /api/patient-history/{patient_id}
Retrieve patient test history

### POST /api/vlm-qa
Ask questions to AI medical assistant

### POST /api/download-pdf/{prediction_id}
Download clinical report PDF

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **GitHub:** [https://github.com/afridpasha/FedFusionNet-](https://github.com/afridpasha/FedFusionNet-)
- **Issues:** [GitHub Issues](https://github.com/afridpasha/FedFusionNet-/issues)

---

## ⚠️ Disclaimer

This application is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

---

<div align="center">

**Made with ❤️ for advancing medical AI and improving patient outcomes**

</div>
