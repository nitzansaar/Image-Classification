# Image Classification Project

This project involves fine-tuning a Vision Transformer (ViT) model on the Food-101 dataset for image classification using the Hugging Face Transformers library.

## Introduction

Image classification is the task of assigning a label or class to an image. This project demonstrates how to:
1. Fine-tune a ViT model on the Food-101 dataset to classify food items in images.
2. Use the fine-tuned model for inference to predict labels for new images.

## Getting Started

### Installation

Before you begin, ensure you have the necessary libraries installed:

```bash
pip install transformers datasets tensorflow keras evaluate
Notebooks and Code
The project code and explanation are available in the Jupyter notebook named image_classification.ipynb.
```
## Steps
- Load a subset of the Food-101 dataset using the Hugging Face Datasets library.
- Define transformations and preprocess the dataset for training and validation.
- Set up a training pipeline using TensorFlow and Hugging Face Transformers.
- Train the model on the dataset and evaluate using the accuracy metric.
- Fine-tune the model using Keras callbacks and push the model to the Hugging Face Hub.
- Perform inference using the fine-tuned model.
## What I Learned
- Loading and preprocessing image datasets using the Hugging Face Datasets library.
- Using Vision Transformer (ViT) models for image classification tasks.
- Applying data augmentation and preprocessing techniques to improve model performance.
- Fine-tuning a ViT model using TensorFlow and Hugging Face Transformers.
- Evaluating the model's accuracy and performance during training.
- Using Keras callbacks to compute metrics and push models to the Hugging Face Hub.
- Performing inference using the fine-tuned model for image classification.