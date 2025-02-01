# **Image Classification using ResNet in PyTorch**

## 📌 Project Overview

This project demonstrates an end-to-end deep learning pipeline for **image classification** using **ResNet-50** in **PyTorch**. The goal is to train a model on a given dataset, evaluate its performance, and perform inference on new images.

### 🔹 Features:
- **Dataset Preprocessing** (Resizing, Normalization, Augmentation)
- **Exploratory Data Analysis (EDA)**
- **ResNet Model Training & Fine-tuning**
- **Model Evaluation** (Accuracy, Loss, and Confusion Matrix)
- **Inference on Unseen Images**

## 🚀 Getting Started

### 🔹 Prerequisites
Make sure you have the following dependencies installed:

- Python (>=3.8)
- PyTorch (>=1.13)
- Torchvision
- Matplotlib
- NumPy

### 🔹 Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/YourGithubName/image-classification-resnet
cd image-classification-resnet
pip install -r requirements.txt

## 📊 Data Preprocessing

The dataset is assumed to be stored in the `data/raw/` directory.

### 🔹 Preprocessing Steps:
- **Resizing images** to 224x224
- **Applying transformations** (Normalization, Random Flipping)
- **Creating train-test split**

### 🖼 Exploratory Data Analysis

- **Visualizing sample images**
- **Checking class distribution**
- **Applying augmentation techniques**

### 🧠 Model Training

- **Uses a pretrained ResNet-50 model
- **Fine-tunes the fully connected layer for custom classification
- **Uses CrossEntropy Loss and Adam Optimizer
- **Runs for 10 epochs with batch size 32

### 📈 Model Evaluation

- **Computes test accuracy
- **Generates loss/accuracy plots
- **Visualizes confusion matrix

### 📌 Inference on New Images

- **Perform classification on a single image

### 🔬 Results

- **Achieved XX% accuracy on the test set

### 📜 License

- This project is licensed under the MIT License.

### 🤝 Contributing
- Feel free to fork this repository and submit pull requests for improvements! 🎯