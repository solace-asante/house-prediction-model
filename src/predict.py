import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from clean import prepare_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"



def predict_house_price(csv_path):
  new_houses = pd.read_csv(csv_path)

  model = joblib.load(MODEL_PATH)

  X_new = prepare_features(new_houses)


  predictions_log = model.predict(X_new)
  predictions = np.expm1(predictions_log)

  results = new_houses.copy()
  results["predicted_price"] = predictions

  return results 

if __name__ == "__main__":
    import sys
    print(predict_house_price(sys.argv[1]))



