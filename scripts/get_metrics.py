import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Need to load some data to test.
# Actually, wait, since we don't have the 3.5M rows split easily, we can just load 10,000 rows.
try:
    df = pd.read_csv("data/kanpur_synthetic_risk_data.csv", nrows=10000)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_crime = joblib.load(os.path.join(BASE_DIR, "src", "models", "xgb_crime.pkl"))
    model_accident = joblib.load(os.path.join(BASE_DIR, "src", "models", "xgb_accident.pkl"))
    model_env = joblib.load(os.path.join(BASE_DIR, "src", "models", "xgb_environment.pkl"))
    model_iso = joblib.load(os.path.join(BASE_DIR, "src", "models", "xgb_isolation.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "src", "models", "feature_columns.pkl"))
    
    X = df[feature_cols]
    
    models = {
        "Crime": (model_crime, "target_crime_risk"),
        "Accident": (model_accident, "target_accident_risk"),
        "Environment": (model_env, "target_environment_risk"),
        "Isolation": (model_iso, "target_isolation_risk"),
    }
    
    print("--- ML EVALUATION METRICS ---")
    for name, (model, target) in models.items():
        y_true = df[target]
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"Model: {name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}\n")
        
except Exception as e:
    print(f"Error: {e}")
