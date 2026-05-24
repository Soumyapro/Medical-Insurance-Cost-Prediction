# 🏥 Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> Predicting medical insurance charges using Machine Learning — comparing **Linear Regression** and **Random Forest Regressor**.

---

## Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models & Results](#-models--results)
- [Visualizations](#-visualizations)
- [Libraries Used](#-libraries-used)

---

## Overview

Medical insurance costs vary significantly based on individual factors such as age, BMI, smoking habits, and region. This project builds and compares two machine learning models to accurately **predict insurance charges** based on these features.

The goal is to find the best performing model that minimises prediction error and maximises accuracy.

---

## Dataset

| Detail | Info |
|---|---|
| **File** | `insurance.csv` |
| **Rows** | 1338 |
| **Target** | `charges` (Insurance cost in USD) |

### Features

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual |
| `sex` | Categorical | Male / Female |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependents |
| `smoker` | Categorical | Yes / No |
| `region` | Categorical | northeast, northwest, southeast, southwest |
| `charges` | Target | Medical insurance cost (USD) |

---

## Project Workflow

```
Raw Data
   │
   ├── Exploratory Data Analysis (EDA)
   │       ├── Distribution plots (categorical)
   │       └── Histogram plots (numerical)
   │
   ├── Feature Engineering
   │       ├── Train-Test Split (80/20)
   │       ├── OneHotEncoding (categorical features)
   │       └── StandardScaler (numerical features)
   │
   └── Model Training & Evaluation
           ├── Linear Regression
           └── Random Forest Regressor
```

---

## Models & Results

### Performance Comparison

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.7836 | $4,181.19 | $5,796.28 |
| **Random Forest** | **0.8653** | **$2,549.00** | **$4,572.84** |

### Winner: Random Forest Regressor

- **R² Score of 0.8653** → explains **86.5%** of variance in charges
- **$1,632 lower MAE** than Linear Regression
- Handles **non-linear relationships** better (e.g. smoker + BMI interaction)

---

## Visualizations

The notebook includes:

- Count plots for categorical features (sex, smoker, region)
- Histograms with KDE for numerical features (age, bmi, children)
- Actual vs Predicted scatter plots for both models

---

## Libraries Used

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
  ├── LinearRegression
  ├── RandomForestRegressor
  ├── OneHotEncoder
  ├── StandardScaler
  ├── ColumnTransformer
  └── train_test_split
```

## Project Structure

```
medical-insurance-cost-prediction/
│
├── medical_insurance_cost_prediction.ipynb   # Main notebook
├── insurance.csv                             # Dataset
└── README.md                                 # Project documentation
```

---

## Key Takeaways

- **Smoking** is the most influential factor in predicting insurance charges
- **Random Forest** significantly outperforms Linear Regression on this dataset due to non-linear feature interactions
- Proper ML pipeline (split → encode → scale → model) is critical to avoid data leakage

---
