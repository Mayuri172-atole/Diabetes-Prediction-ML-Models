# Diabetes-Prediction-ML-Models
A comprehensive machine learning project to predict diabetes using multiple algorithms, including Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Support Vector Machines, and Artificial Neural Networks, with performance comparison and a high-accuracy hybrid model.


# Diabetes Prediction using Machine Learning and Deep Learning

This repository contains a complete pipeline for predicting diabetes based on patient health metrics using various Machine Learning (ML) and Deep Learning (DL) models. The project evaluates multiple algorithms and introduces a hybrid model that achieves **95.5% accuracy**.

 Features
- **Data Preprocessing**
  - Missing value handling
  - Feature scaling
  - Exploratory Data Analysis (EDA)
- **Machine Learning Models**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Gaussian & Bernoulli)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
  - Support Vector Machine (SVM)
  - AdaBoost
  - Bagging Classifier
- **Deep Learning**
  - Artificial Neural Network (ANN)
  - Optimized ANN with Dropout & Batch Normalization
- **Hybrid Model**
  - Combines Random Forest and ANN outputs using XGBoost meta-learner
  - Achieves **0.955 accuracy score**
- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix & Visualization

 Dataset
- Processed dataset: `processed_diabetes.csv`
- Features:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target: 0 = No Diabetes, 1 = Diabetes)

 Model Performance Summary
| Model                   | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Random Forest           | 0.845    | 0.805     | 0.91   | 0.854    |
| CatBoost                | 0.840    | 0.798     | 0.91   | 0.850    |
| LightGBM                | 0.835    | 0.819     | 0.86   | 0.839    |
| Support Vector Machine  | 0.825    | 0.792     | 0.88   | 0.834    |
| **Hybrid Model**        | **0.955**| **0.940** | **0.970**| **0.960**|

Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Diabetes-Prediction-ML-Models.git
   cd Diabetes-Prediction-ML-Models

    Install dependencies:

pip install -r requirements.txt

Open the Jupyter Notebook or Google Colab file:

    jupyter notebook Diabetes.ipynb

    Run all cells to train and evaluate models.

 Requirements

    Python 3.8+

    pandas

    numpy

    matplotlib

    seaborn

    scikit-learn

    xgboost

    lightgbm

    catboost

    tensorflow / keras

Results

The Hybrid Model outperforms all individual models, making it a strong candidate for real-world diabetes prediction tasks.
