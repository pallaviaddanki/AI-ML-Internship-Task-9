# AI-ML-Internship-Task-9

# Credit Card Fraud Detection using Random Forest

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Since fraud datasets are highly imbalanced, accuracy alone is not a reliable metric. This project uses precision, recall, and F1-score to properly evaluate model performance.

A baseline Logistic Regression model is compared with a Random Forest ensemble model to show the effectiveness of ensemble learning for fraud detection.

## Dataset
Primary Dataset: Kaggle Credit Card Fraud Dataset  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

- Total Transactions: 284,807  
- Fraud Cases: 492  
- Non-Fraud Cases: 284,315  
- Target Column: Class  
  - 0 = Non-Fraud  
  - 1 = Fraud  

If the dataset is not uploaded to this repository due to size limits, it can be downloaded from the above link.

## Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  
- Jupyter Notebook  

## Project Workflow
1. Loaded the dataset and analyzed class imbalance.
2. Cleaned the data and handled missing values.
3. Separated features and target variable.
4. Used stratified train-test split.
5. Trained a baseline Logistic Regression model.
6. Trained a Random Forest classifier.
7. Evaluated models using precision, recall, and F1-score.
8. Plotted feature importance.
9. Saved the trained model using Joblib.

## Models Used
- Logistic Regression (Baseline)
- Random Forest Classifier (Final Model)

Random Forest Parameters:
- n_estimators = 100
- class_weight = balanced

## Evaluation Metrics
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics are used instead of accuracy due to class imbalance.

## Feature Importance
Feature importance from the Random Forest model was plotted to identify key features contributing to fraud detection.

## Saved Model
The trained model is saved as:

random_forest_fraud_model.pkl

## Repository Structure
credit-card-fraud-rf/
├── fraud_detection_rf.ipynb
├── random_forest_fraud_model.pkl
├── feature_importance.png
├── README.md
└── creditcard.csv (or dataset download link)

## Key Learnings
- Handling imbalanced datasets
- Importance of proper evaluation metrics
- Ensemble learning concepts
- Random Forest for fraud detection
- Saving and loading ML models

What is SMOTE?  
Synthetic Minority Oversampling Technique used to generate synthetic samples for the minority class.
