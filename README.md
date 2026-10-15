# House Price Prediction

A machine learning project that predicts house sale prices from property features using regression.

## Overview

This project explores the relationship between property characteristics and sale prices through data analysis, feature engineering, model experimentation, evaluation, and prediction.

## Dataset

The dataset contains house-sale records with features including bedrooms, bathrooms, living area, lot size, construction grade, building age, renovation history, and geographical coordinates.

**Source:** [Predicting House Price — Kaggle](https://www.kaggle.com/datasets/raotariq/predicting-house-price-dataset/data)

## Methodology

### Data Cleaning

I cleaned the raw dataset and created a reusable Python function for consistent feature preparation. The cleaned data was saved as `clean.csv`.

### Exploratory Data Analysis

I explored feature distributions, correlations, and potential multicollinearity using Jupyter Notebook.

### Feature Engineering

I applied `log1p` transformations to `sqft_lot` and `sqft_lot15`, created a binary `renovated` feature, handled unusually high bedroom counts, and selected 16 features for modelling. The target variable, `price`, was also log-transformed.

### Model Development

I experimented with baseline linear regression, feature ablation, Ridge regression, Lasso regression, and polynomial regression. Based on the experiments, I selected a pipeline combining median imputation, feature scaling, second-degree polynomial features, and linear regression.

| Model | Validation RMSE (log price) |
|---|---|
| Baseline predict the mean | 0.526 |
| Linear regression | about 0.253 |
| Ridge / Lasso | about 0.253|
| Polynomial, degree 2 | 0.225 |

Hold-out test (notebook, used once): RMSE 0.227 on log price, MAE about $96k.

### Model Evaluation

I used five-fold cross-validation with `GroupKFold` and evaluated performance using RMSE on the log-transformed target.

| Metric | RMSE |
|---|---:|
| Mean training RMSE | 0.222 |
| Mean validation RMSE | 0.225 |

The similar scores indicate comparable training and validation performance across the evaluated folds.

### Model Saving and Prediction

After evaluation, I fitted the pipeline on the full dataset and saved the trained model using `joblib`. A reusable prediction function loads new house data from a CSV file, prepares the features, generates predictions, converts them back to the original price scale, and returns the original data with predicted prices.

## Challenges and Problem-Solving

Maintaining consistent features between training and prediction was an important challenge. I addressed this by creating a reusable feature-preparation function that returns only the 16 intended model features and separating data cleaning, training, and prediction into individual scripts.

## Limitations

**Geographic scope:** The model uses historical house-sale data and has not been validated for other housing markets.

**Prediction accuracy:** RMSE is calculated on the log-transformed target and does not directly represent errors in dollars.

**Feature limitations:** The model cannot account for factors absent from the dataset, including unrecorded property characteristics and changing market conditions.

**Validation limitations:** Cross-validation estimates performance on held-out data but does not guarantee accuracy on future sales.

**Model limitations:** Polynomial linear regression may not capture every complex relationship between property features and sale prices.

## Future Improvements

Try random forest and XGBoost, test PCA, and serve predictions with a FastAPI endpoint.

## How to run
1. Download housing.csv from Kaggle and place it at data/raw/housing.csv
2. python -m venv .venv && source .venv/bin/activate
3. pip install -r requirements.txt
4. python src/train.py   (prints CV scores, saves models/house_price_model.joblib)
5. 5. Predict from Python:
   import sys; sys.path.append("src")
   from predict import predict_house_price
   predict_house_price("path/to/new_houses.csv")

## Project Structure

```text
house-prediction-model/
├── data/
│   ├── raw/
│   └── processed/
│       └── clean.csv
├── models/
│   └── house_price_model.joblib
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_MODELLING.ipynb
├── src/
│   ├── clean.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

Python, pandas, NumPy, scikit-learn and Jupyter Notebook.

## What I Learned

I gained practical experience in exploratory data analysis, feature engineering, regression modelling, feature selection, cross-validation, model evaluation, saving trained models, and generating predictions on new data.

## Disclaimer

Predictions are estimates based on historical data and should not be used as the sole basis for property purchasing or financial decisions.
