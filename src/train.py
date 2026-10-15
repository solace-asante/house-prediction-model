from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate,GroupKFold

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

from sklearn.linear_model import LinearRegression

import joblib
from clean import prepare_features

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "housing.csv"

raw = pd.read_csv(RAW)

X = prepare_features(raw)
y = np.log1p(raw["price"])
groups = raw['id']


model = make_pipeline(
  SimpleImputer(strategy="median"),
  StandardScaler(),
  PolynomialFeatures(degree=2),
  LinearRegression()

)

validation = cross_validate(
  model,X,y,
  cv=(GroupKFold(n_splits=5)),
  groups=groups,
  scoring="neg_root_mean_squared_error",
  return_train_score=True

)
print("Train RMSE:", -validation ["train_score"].mean())

print("Validation RMSE:", -validation ["test_score"].mean())


model.fit(X,y)

MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)

joblib.dump(model,MODEL_PATH)
print(f"Model saved to: {MODEL_PATH} successfully")


