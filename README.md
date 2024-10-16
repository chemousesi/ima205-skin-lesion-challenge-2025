# AlexNet for Skin Lesion Classification

This repository contains the code for the project **Fine-Tuning and Boosting AlexNet for Skin Lesion Classification**. The primary objective is to classify skin lesions using an AlexNet-based deep learning architecture, fine-tuned with physical data augmentation techniques to enhance performance.

link to the competition : https://www.kaggle.com/competitions/ima205-challenge-2024/leaderboard
Detailed report is in : IMA_205_Project_report_AlexNet_for_Skin_lesion_classification

## Overview

Skin cancer, particularly malignant melanoma, is a significant health concern, accounting for the majority of skin cancer-related deaths. Early detection is crucial to improve patient outcomes. This project leverages a deep learning model, AlexNet, to classify different types of skin lesions from the **ISIC** dataset. The project involved extensive data preprocessing, augmentation, and fine-tuning of a pretrained AlexNet model to optimize classification performance.

## Project Goals

- Utilize the **ISIC dataset** to train and fine-tune a neural network for skin lesion classification.
- Apply various data augmentation techniques to improve model generalization.
- Achieve efficient and accurate classification while minimizing computational resources.

## Dataset

- **Training Set**: 18,998 images
- **Test Set**: 6,333 images
- **Classes**: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion, Squamous cell carcinoma

## Key Techniques

### Data Preprocessing

- **Hair Removal**: Morphological closing and hysteresis thresholding to remove occlusions.
- **Data Augmentation**: Various transformations such as flipping, rotation, brightness adjustment, noise addition, and more to prevent overfitting and enhance model performance.

### Model Architecture

We used a modified AlexNet architecture for classification with 8 output classes. The architecture includes:

- Convolutional layers with ReLU activation
- Max Pooling
- Fully connected layers
- Softmax output for classification

### Evaluation

The primary metric used for evaluation was **Weighted Categorization Accuracy (WA)**, which considers class imbalance in the dataset.

## Results

- Initial model accuracy without augmentation: **< 50%**
- Final model accuracy after augmentation: **69%**
- Ranking: **13th out of 80** in a school Kaggle competition

## Challenges

- Integration of metadata with image data.
- Time constraints and limited access to resources.
- Handling dataset instability due to the use of patches.

## Future Work

- Experimenting with alternative architectures like multi-class SVMs.
- Incorporating more metadata for further performance improvements.

## Requirements

- Python 3.12
- Libraries: 
  - `torch`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `albumentations` for data augmentation

Install dependencies using:

```bash
pip install -r requirements.txt

