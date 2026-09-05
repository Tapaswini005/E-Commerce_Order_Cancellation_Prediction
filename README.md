# E-Commerce Order Cancellation Prediction

## 📌 Project Overview

This project predicts whether an e-commerce order is likely to be cancelled based on order and customer-related information.

The project uses Machine Learning classification algorithms to learn patterns from historical order data and predict cancellation for new orders.

## 🎯 Objective

The main objective of this project is to build a simple and reliable machine learning system that can:

* Analyze historical e-commerce order data
* Clean and preprocess the data
* Perform Exploratory Data Analysis (EDA)
* Train multiple classification models
* Compare model performance
* Predict whether a new order will be cancelled

## 🤖 Machine Learning Models

The following models will be trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction
   ↓
Streamlit Application
```

## 📂 Project Structure

```text
ecommerce_order_cancellation_prediction/
│
├── data/
│   ├── raw/
│   │   └── orders.csv
│   └── processed/
│       └── processed_data.csv
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── visualization/
│   │   └── eda.py
│   │
│   └── models/
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset contains historical e-commerce order information.

Possible features include:

* Customer information
* Order value
* Product category
* Payment method
* Discount
* Delivery information
* Previous order information
* Previous cancellation information

The target variable is:

```text
Cancelled
```

where:

```text
0 → Order not cancelled
1 → Order cancelled
```

## 📈 Model Evaluation

The trained models will be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The model with the best overall performance will be selected for prediction.

## 🖥️ Application

A simple Streamlit application will allow the user to enter the details of a new order and receive a cancellation prediction.

Example:

```text
Order Details
      ↓
Machine Learning Model
      ↓
Cancellation Prediction
      ↓
Cancelled / Not Cancelled
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

## 🚀 Project Status

Currently under development.