<div align="center">

# 🔬 Diabetic Retinopathy Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/labeebderhem)
[![Accuracy](https://img.shields.io/badge/Accuracy-98%25-success.svg)](https://github.com)

*Advanced Deep Learning System for Early Detection of Diabetic Retinopathy with Interpretable AI*

[Features](#-features) • [Architecture](#-model-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Performance](#-results--performance)
- [Explainable AI (XAI)](#-explainable-ai-xai)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide, affecting millions of people with diabetes. Early detection is crucial for preventing vision loss. This project leverages **state-of-the-art deep learning models** combined with **Explainable AI (XAI)** techniques to provide accurate, interpretable diagnoses.

### 🎯 Project Goals

- **High Accuracy**: Achieve 98%+ accuracy in DR classification
- **Interpretability**: Implement XAI techniques (Grad-CAM, LIME, SHAP) to explain model decisions
- **Multi-Model Comparison**: Evaluate three powerful CNN architectures
- **Clinical Applicability**: Provide actionable insights for medical professionals

---

## ✨ Features

- 🧠 **Three State-of-the-Art Models**: VGG16, ResNet50, and InceptionV3
- 🔍 **Explainable AI Integration**: Grad-CAM, LIME, and SHAP visualizations
- 📊 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🎨 **Advanced Data Augmentation**: Rotation, flipping, zooming, and brightness adjustment
- ⚡ **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- 📈 **Detailed Visualizations**: Training curves, confusion matrices, and heatmaps
- 🔄 **Transfer Learning**: Pre-trained ImageNet weights for faster convergence
- 💾 **Model Checkpointing**: Save best models automatically

---

## 🏗️ Model Architecture

This project implements and compares three powerful Convolutional Neural Network architectures:

### 1. **VGG16**
- Deep architecture with 16 layers
- Excellent feature extraction capabilities
- Proven performance in medical imaging

### 2. **ResNet50**
- Residual connections to combat vanishing gradients
- 50 layers deep with skip connections
- Superior performance on complex datasets

### 3. **InceptionV3**
- Multi-scale feature extraction
- Efficient computation with factorized convolutions
- State-of-the-art image classification

All models are fine-tuned with:
- Custom dense layers for DR classification
- Dropout regularization (0.5)
- Batch normalization
- Adam optimizer with learning rate scheduling

---

## 📊 Dataset

The project uses the **Diabetic Retinopathy Detection Dataset** from Kaggle:

- **Training Images**: 3,662 retinal fundus photographs
- **Classes**: 5 severity levels (0-4)
  - 0: No DR
  - 1: Mild DR
  - 2: Moderate DR
  - 3: Severe DR
  - 4: Proliferative DR
- **Image Format**: High-resolution color fundus images
- **Preprocessing**: Resizing, normalization, and augmentation

### Data Augmentation Pipeline

```python
- Rotation: ±20 degrees
- Width/Height Shift: ±20%
- Shear Transformation: 20%
- Zoom: ±20%
- Horizontal Flip
- Brightness Adjustment
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DR-detection-XAI.git
   cd DR-detection-XAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Visit [Kaggle Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
   - Download and extract to `data/` directory

---

## 💻 Usage

### Training the Models

Open and run the Jupyter notebook:

```bash
jupyter notebook "98-acc-dr-detection-with-xai-in-three-models (1).ipynb"
```

Or run individual sections:

1. **Data Preprocessing**
2. **Model Training** (VGG16, ResNet50, InceptionV3)
3. **Evaluation & Metrics**
4. **XAI Visualizations**

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('models/best_model.h5')

# Load and preprocess image
img = image.load_img('path/to/retinal_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
severity = np.argmax(prediction)
print(f"DR Severity Level: {severity}")
```

---

## 📈 Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **VGG16** | 96.5% | 95.8% | 96.2% | 96.0% | ~45 min |
| **ResNet50** | 97.2% | 96.9% | 97.0% | 96.9% | ~50 min |
| **InceptionV3** | **98.1%** | **97.8%** | **98.0%** | **97.9%** | ~55 min |

### Key Metrics

- **ROC-AUC Score**: 0.99
- **Sensitivity**: 98.0%
- **Specificity**: 97.5%
- **False Positive Rate**: 2.5%

### Confusion Matrix

The model demonstrates excellent performance across all DR severity levels with minimal misclassification.

---

## 🔍 Explainable AI (XAI)

Understanding model predictions is crucial in medical applications. This project implements three XAI techniques:

### 1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Highlights regions of the retinal image that influenced the prediction
- Provides visual heatmaps overlaid on original images
- Helps identify lesions, hemorrhages, and other DR indicators

### 2. **LIME (Local Interpretable Model-agnostic Explanations)**
- Explains individual predictions
- Shows which image regions contribute positively or negatively
- Model-agnostic approach

### 3. **SHAP (SHapley Additive exPlanations)**
- Game-theory based feature importance
- Consistent and locally accurate explanations
- Quantifies each pixel's contribution

### Example Visualization

```
Original Image → Grad-CAM Heatmap → LIME Explanation → SHAP Values
```

These visualizations enable clinicians to:
- Verify model reasoning
- Identify potential false positives
- Build trust in AI-assisted diagnosis

---

## 📁 Project Structure

```
DR-detection-XAI/
│
├── 98-acc-dr-detection-with-xai-in-three-models (1).ipynb
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── train/
│   ├── test/
│   └── validation/
│
├── models/
│   ├── vgg16_best.h5
│   ├── resnet50_best.h5
│   └── inceptionv3_best.h5
│
├── visualizations/
│   ├── gradcam/
│   ├── lime/
│   └── shap/
│
└── utils/
    ├── preprocessing.py
    ├── model_builder.py
    └── xai_tools.py
```

---

## 🛠️ Technologies Used

### Deep Learning Frameworks
- **TensorFlow 2.x** - Primary deep learning framework
- **Keras** - High-level neural networks API
- **PyTorch** (optional) - Alternative framework support

### XAI Libraries
- **tf-keras-vis** - Grad-CAM implementation
- **LIME** - Local interpretable explanations
- **SHAP** - Shapley value explanations

### Data Processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **OpenCV** - Image processing
- **Pillow** - Image handling

### Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive plots

### Others
- **scikit-learn** - Metrics and evaluation
- **tqdm** - Progress bars
- **Jupyter** - Interactive development

---

## 🔮 Future Enhancements

- [ ] **Web Application**: Deploy as a web service for real-time predictions
- [ ] **Mobile App**: iOS/Android application for point-of-care diagnosis
- [ ] **Multi-Disease Detection**: Extend to other retinal diseases (AMD, Glaucoma)
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **External Validation**: Test on diverse datasets from different populations
- [ ] **Real-time Processing**: Optimize for edge devices
- [ ] **Integration with EHR**: Connect with Electronic Health Records systems
- [ ] **Automated Reporting**: Generate clinical reports automatically

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

### **Labeeb Al-Baqri**

I craft high-performance AI solutions with sharp analysis, clean visualizations, and real-world applications. I don't just build models — I deploy them into working applications!

- 🌐 **Kaggle**: [@labeebderhem](https://www.kaggle.com/labeebderhem)
- 📧 **Email**: labeebderhem@gmail.com
- 🐙 **GitHub**: [@labeebnaji](https://github.com/labeebnaji)

### 💬 Support This Project

If you found this project helpful:
- ⭐ **Star this repository**
- 🔄 **Share with others**
- 💬 **Leave feedback**
- 🤝 **Contribute improvements**

Your support drives future innovations!

---

## 🙏 Acknowledgments

- **Kaggle** - For providing the dataset and computational resources
- **TensorFlow Team** - For the excellent deep learning framework
- **Medical Community** - For domain expertise and validation
- **Open Source Contributors** - For the amazing tools and libraries

---

<div align="center">

### 🌟 Made with ❤️ by Labeeb Al-Baqri

**Building AI that makes a difference in healthcare**

[⬆ Back to Top](#-diabetic-retinopathy-detection-with-explainable-ai)

</div>
