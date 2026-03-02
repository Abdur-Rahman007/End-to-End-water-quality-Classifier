# Water Quality (Potability) Predictor — End-to-End ML + Gradio App

An end-to-end Machine Learning project that predicts **water potability (0 = Not Potable, 1 = Potable)** using a **Scikit-learn Pipeline** and a **Gradio web interface**.  
The project includes data preprocessing, cross-validation, hyperparameter tuning using GridSearchCV, model evaluation, model persistence (`.pkl`), and deployment through a simple UI.

---

## Project Overview

This repository contains:

- **Model training script** that:
  - Loads the dataset
  - Builds a preprocessing + model pipeline
  - Runs Cross-Validation
  - Performs Hyperparameter Tuning (GridSearchCV)
  - Evaluates performance (Accuracy, Classification Report, Confusion Matrix)
  - Saves the final best model pipeline to `water_rf_pipeline.pkl`  
- **Gradio app** that:
  - Loads the saved pipeline
  - Takes user inputs for water features
  - Predicts the potability result

---

## Dataset

Expected CSV file: `water_potability.csv`

Target column:
- `Potability` (binary: 0/1)

Features used in the app:
- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

---

## Model & ML Pipeline

The training script uses a **Pipeline** containing:

1. **Median Imputation** to handle missing values  
2. **Standard Scaling** to normalize numeric feature scales  
3. **RandomForestClassifier** as the primary model  

## Model Training & Evaluation

- **Train–Test Split:** 80% / 20%
- **Cross-Validation:** 7-fold CV using Accuracy
- **Hyperparameter Tuning:** GridSearchCV (scored using F1-score)
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
    
Hyperparameter tuning is performed using **GridSearchCV** and scored using **F1-score**.  
The best estimator is then selected and saved as `water_rf_pipeline.pkl`.

---

## Repository Structure

```bash
.
├── water_prediction.py        # Model training, tuning, evaluation, saving pipeline
├── app.py                     # Gradio web application
├── water_potability.csv       # Dataset (add manually)
├── water_rf_pipeline.pkl      # Saved trained ML pipeline (generated)
├── requirements.txt           # Project dependencies
└── README.md
