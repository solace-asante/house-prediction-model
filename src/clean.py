import numpy as np
import pandas as pd
from pathlib import Path


def prepare_features(df):
  out = df.copy()
  out.columns = out.columns.str.strip()


  required_columns= [
      "bedrooms",
      "bathrooms",
      "sqft_lot",
      "floors",
      "waterfront",
      "view",
      "condition",
      "grade",
      "sqft_above",
      "sqft_basement",
      "yr_built",
      "yr_renovated",
      "lat",
      "long",
      "sqft_living15",
      "sqft_lot15",
  ]
  missing = set(required_columns) - set(out.columns)

  if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

  out['sqft_lot'] = np.log1p(out['sqft_lot'])
  out['sqft_lot15'] = np.log1p(out['sqft_lot15'])

  
  out['renovated'] = (out['yr_renovated'] > 0).astype(int)
  out.loc[out['bedrooms'] > 20, 'bedrooms'] = np.nan
  feature_columns = [
    "bedrooms",
    "bathrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "renovated",
]
  out = out[feature_columns]

  return out 

