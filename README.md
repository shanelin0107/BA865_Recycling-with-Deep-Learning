# BA865_Recycling-with-Deep-Learning
# Recycling with Deep Learning: A CNN-Based Approach to Waste Classification

## Table of Contents
- [Overview](#overview)  
- [Motivation](#motivation)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Model Architecture](#model-architecture)  
  - [Custom CNN](#custom-cnn)  
  - [Transfer Learning](#transfer-learning)  
- [Model Evaluation](#model-evaluation)  
- [Conclusion](#conclusion)  
- [Challenges](#challenges)  
- [Future Work](#future-work)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Running the Notebook](#running-the-notebook)  
- [Authors](#authors)  
- [License](#license)  

---

## Overview
Landfills occupy over 1.8 million acres in the U.S. and rank as the third-largest human-related methane source. Despite a 2030 recycling goal of 50%, only 32.1% of municipal solid waste is recycled—often because disposal is cheaper. We propose a CNN-based image classifier to sort waste into six categories (plastic, general waste, glass, organic, metal, paper) and integrate it into smart bins to boost recycling rates and cut costs. :contentReference[oaicite:0]{index=0}

## Motivation
Inefficient sorting raises both environmental and municipal costs, and public confusion exacerbates low diversion rates. Automating classification with a fine-tuned CNN embedded in waste bins can:
- Help governments meet environmental targets  
- Enable waste firms to reduce sorting labor  
- Guide households toward correct recycling habits :contentReference[oaicite:1]{index=1}

## Dataset
- **Custom images** of everyday trash items, manually labeled into six classes.  
- **Splits**: Training, validation, and test sets with a balanced distribution across categories. :contentReference[oaicite:2]{index=2}

## Preprocessing
1. **Resize & rescale** all images to 224×224 pixels, normalize pixel values to [0,1].  
2. **Augment** with horizontal flips, random rotations (±10%), and zooms (±20%) to improve robustness. :contentReference[oaicite:3]{index=3}

## Model Architecture

### Custom CNN
- **Layers**:  
  - Augmentation pipeline → Conv2D (32→64→128) + MaxPooling → Flatten → Dense(6, softmax)  
- **Config**:  
  - Optimizer: RMSprop  
  - Loss: sparse categorical crossentropy  
  - Experiments: added Dense(256), Dropout(30%), BatchNorm, brightness/contrast adjustments :contentReference[oaicite:4]{index=4}

### Transfer Learning
- **ResNet50**  
  - Frozen backbone, custom head: GlobalAveragePooling2D → Dense(128) → Softmax  
  - Result: poor accuracy (~28%) due to overfitting on small data :contentReference[oaicite:5]{index=5}
- **MobileNetV2 + CutMix**  
  - Lightweight base → custom Conv/Pooling blocks → CutMix augmentation  
  - Achieved highest validation accuracy and smoother convergence :contentReference[oaicite:6]{index=6}

## Model Evaluation
- **Baseline CNN**: ~70.8% accuracy after tuning (extra dense layer, dropout, advanced augmentations).  
- **ResNet50**: Test accuracy ~28.3%, test loss ~1.59.  
- **MobileNetV2 + CutMix**: Best trade-off of size and accuracy, ultimately achieving ~83% on six classes. :contentReference[oaicite:7]{index=7}

## Conclusion
A MobileNetV2-based classifier with CutMix can deliver real-time, accurate sorting in resource-constrained settings, accelerating recycling operations, lowering costs, and cutting greenhouse-gas emissions. :contentReference[oaicite:8]{index=8}

## Challenges
- **Mixed formats**: JPEG, PNG, HEIC introduced compression artifacts.  
- **Single-label design**: Multi-item photos confused the model.  
- **Limited data**: Small dataset hindered larger architectures. :contentReference[oaicite:9]{index=9}

## Future Work
- Multi-object detection for mixed-item scenes  
- Real-time edge deployment (e.g., Raspberry Pi + OpenCV)  
- Expanded, diverse dataset for better generalization  
- Experiment with vision transformers (ViT, DeiT)  
- Model ensembles (MobileNetV2 + ResNet50 + EfficientNet) :contentReference[oaicite:10]{index=10}

## Getting Started

### Prerequisites
- Python 3.8+  
- TensorFlow 2.x  
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`  
- Jupyter Notebook  

### Installation
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
