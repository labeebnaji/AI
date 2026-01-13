<div align="center">

# 🔬 Diabetic Retinopathy Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/labeebderhem)

# Diabetic Retinopathy Detection using Deep Learning with Explainable AI

This project implements an automated system for detecting Diabetic Retinopathy (DR) from retinal fundus images using three state-of-the-art CNN architectures, with Explainable AI techniques to interpret model decisions.

Diabetic Retinopathy is the leading cause of blindness in working-age adults, affecting over 93 million people worldwide. Early detection is crucial but manual screening is time-consuming and requires trained specialists. This project aims to provide an accurate, interpretable automated screening solution.

## Models Used

Three pretrained CNN architectures with custom classification heads:

- **EfficientNetB3** - Input: 300×300, efficient compound scaling
- **DenseNet201** - Input: 224×224, dense connections for feature reuse
- **ResNet50V2** - Input: 224×224, residual learning with skip connections

All models use Transfer Learning from ImageNet weights with two-phase training:
1. Frozen base model training
2. Fine-tuning with partial unfreezing

## Results

### Test Set Performance (Same Distribution)

| Model | Accuracy | AUC | Loss |
|-------|----------|-----|------|
| DenseNet201 | 96.55% | 0.994 | 0.092 |
| ResNet50V2 | 96.18% | 0.992 | 0.117 |
| EfficientNetB3 | 94.00% | 0.988 | 0.169 |

### External Validation (APTOS 2019 Dataset)

| Model | Accuracy | AUC | F1-Score (DR) |
|-------|----------|-----|---------------|
| DenseNet201 | 92.05% | 0.985 | 0.93 |
| EfficientNetB3 | 84.77% | 0.968 | 0.87 |
| ResNet50V2 | 77.27% | 0.921 | 0.82 |

**Best Model:** DenseNet201 achieved the highest performance on both internal test set and external validation, demonstrating strong generalization capability. Its dense connectivity pattern enables better feature extraction for medical imaging tasks.

## Explainable AI (XAI)

Two visualization techniques are implemented to interpret model predictions:

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Computes gradients flowing into the final convolutional layer
- Generates heatmaps highlighting regions influencing the prediction
- Helps verify that models focus on clinically relevant features (lesions, hemorrhages)

### Score-CAM
- Gradient-free alternative using activation map contributions
- More stable visualizations across different inputs
- Better suited for comparing explanations across models

These XAI techniques are essential for building trust with clinicians and validating that the model learns meaningful patterns rather than artifacts.

## Dataset

- **Training/Validation/Test:** Diabetic Retinopathy 224×224 Gaussian Filtered Images
- **External Validation:** APTOS 2019 Blindness Detection Challenge
- **Split:** 70% Train / 15% Validation / 15% Test (Stratified)
- **Classification:** Binary (DR vs No_DR)


## Usage

1. Run the notebook on Kaggle with GPU enabled
2. Models are saved as `.model` files after training
3. Use `predict_class()` function for inference on new images

## Author

### **Labeeb Al-Baqeri**

I craft high-performance AI solutions with sharp analysis, clean visualizations, and real-world applications. I don't just build models — I deploy them into working applications!

- 🌐 **Kaggle**: [@labeebderhem](https://www.kaggle.com/labeebderhem)
- 📧 **Email**: labeebderhem@gmail.com
- 🐙 **GitHub**: [@labeebnaji](https://github.com/labeebnaji)