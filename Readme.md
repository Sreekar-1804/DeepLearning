# Explainable Age and Gender Prediction using Deep Learning

This project is an end-to-end deep learning application for predicting **gender** and **age group** from face images using a **multi-task ResNet34 model**. The project includes data preprocessing, transfer learning, model evaluation, prediction confidence scores, Grad-CAM explainability, and a Streamlit web demo.

The goal of this project is not only to classify facial attributes, but also to make the model more interpretable using Grad-CAM visualizations.

---

## Project Overview

The model predicts two outputs from a single face image:

1. **Gender Prediction**
   - Female
   - Male

2. **Age Group Prediction**
   - Child
   - Young Adult
   - Adult
   - Senior

Instead of predicting exact age, the project predicts age groups. This is a more practical approach because exact age prediction is harder, noisier, and more error-prone.

---

## Key Features

- Image upload through a Streamlit web app
- Multi-task learning model with one shared ResNet34 backbone
- Gender and age group prediction from a single image
- Confidence scores for both predictions
- Grad-CAM heatmaps for explainability
- Validation metrics including accuracy, precision, recall, F1-score, and confusion matrix
- Error analysis for incorrect predictions
- Unlabeled test image prediction pipeline
- Modular project structure for training, inference, and deployment

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet34
- Transfer Learning
- Streamlit
- Grad-CAM
- OpenCV
- PIL
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Model Architecture

This project uses a **multi-task learning architecture**.

Input Face Image
       |
       v
Preprocessing
       |
       v
ResNet34 Backbone
       |
       v
Shared Feature Vector
       |
       |----------------------|
       v                      v
Gender Head              Age Group Head
Female / Male            Child / Young Adult / Adult / Senior

The ResNet34 backbone extracts shared visual features from the face image. Two separate classification heads then use these features to predict gender and age group.
--- 

## Why Multi-task Learning?

Gender and age group prediction both depend on facial features. Instead of training two separate models, a shared model can learn common visual patterns and use separate output heads for each task.

This improves project structure and makes the pipeline more efficient.

## Why Age Groups Instead of Exact Age?

Exact age prediction is difficult because:

people of the same age can look very different
lighting, pose, and image quality affect predictions
age labels can be noisy
exact age is harder to evaluate reliably

Therefore, this project converts exact age into four age groups:

Age Range	Label
0–12	Child
13–30	Young Adult
31–55	Adult
56+	Senior
Dataset Structure

The dataset contains training images, test images, and CSV files.

face_dataset/
│
├── train/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
├── test/
│   ├── image_101.jpg
│   ├── image_102.jpg
│   └── ...
│
├── train.csv
├── test.csv
├── train_clean.csv
├── val_clean.csv
└── test_clean.csv

The original train.csv contains image paths, age, and gender labels. Since the original test.csv does not contain age or gender labels, the training data was split into:

training set
validation set

The unlabeled test set is used only for final predictions.

## Data Preprocessing

The preprocessing pipeline includes:

image loading using PIL
RGB conversion
resizing to 224 x 224
data augmentation for training
ImageNet normalization
train-validation split
age-to-age-group conversion
gender label encoding
image path validation

## Training transformations:

Resize
Random Horizontal Flip
Random Rotation
Color Jitter
ToTensor
ImageNet Normalization

## Validation and inference transformations:

Resize
ToTensor
ImageNet Normalization

Random augmentation is not applied during validation or inference.

## Training Pipeline

The training pipeline includes:

Load and clean CSV files
Validate image paths
Convert age into age groups
Encode gender labels
Create PyTorch Dataset and DataLoader
Build multi-task ResNet34 model
Train using two classification losses
Validate after every epoch
Save the best model checkpoint
Plot training and validation performance

The final training loss is calculated as:

total_loss = gender_loss + age_group_loss

Both losses use CrossEntropyLoss.

## Evaluation

The model is evaluated separately for both tasks.

Evaluation metrics:

Accuracy
Precision
Recall
F1-score
Confusion matrix

The validation set is used for evaluation because the test set does not contain labels.

## Example output format:

Gender Accuracy: 0.XX
Age Group Accuracy: 0.XX
Average Validation Score: 0.XX

Add your actual results here:

Task	Accuracy	Precision	Recall	F1-score
Gender Prediction	XX%	XX%	XX%	XX%
Age Group Prediction	XX%	XX%	XX%	XX%
Grad-CAM Explainability

Grad-CAM is used to visualize which regions of the face influenced the model prediction.

The project generates separate Grad-CAM heatmaps for:

gender prediction
age group prediction

Grad-CAM helps inspect whether the model focuses on relevant facial regions instead of unrelated background areas.

## Important limitation:

Grad-CAM does not prove the model is correct. It only shows which image regions had strong influence on the prediction.

Streamlit Web Demo

The project includes a Streamlit app where users can upload a face image and get:

predicted gender
gender confidence
predicted age group
age group confidence
gender Grad-CAM heatmap
age group Grad-CAM heatmap

Run the app using:

streamlit run app.py

or:

python -m streamlit run app.py

## Final Project Structure
face_dataset/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── model_loader.py
│   ├── multitask_resnet34.pth
│   └── __init__.py
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── gradcam.py
│   └── __init__.py
│
├── outputs/
│   ├── reports/
│   │   ├── training_history.csv
│   │   ├── validation_predictions.csv
│   │   ├── metrics_summary.csv
│   │   └── test_predictions.csv
│   │
│   ├── plots/
│   └── gradcam_examples/
│
├── sample_images/
│
├── demo_screenshots/
│
├── train/
├── test/
├── train.csv
├── test.csv
├── train_clean.csv
├── val_clean.csv
└── test_clean.csv

## Author
### Naga Sai Satya Sreekar Vanka
LinkedIn: www.linkedin.com/in/sreekar-v/
GitHub: https://github.com/Sreekar-1804
