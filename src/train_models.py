import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import os

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
print("Loading dataset (this may take a moment)...")
start_time = time.time()
df = pd.read_csv("kanpur_synthetic_dataset.csv")
print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")

# Define our independent variables (features)
# We exclude 'h3_cell' and 'timestamp' as they are metadata, not ML features.
features = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_festival_day',
    'cctv_coverage_score', 'tourist_footfall_density', 
    'crime_count_historical', 'reported_harassment_cases', 'recent_crime_count',
    'night_crime_ratio', 'avg_crime_severity', 'traffic_density',
    'road_hazard_score', 'road_lighting_score', 
    'aqi', 'rainfall_mm', 'flood_risk_score', 'police_distance_km',
    'hospital_distance_km', 'ambulance_reach_time_mins', 'network_score'
]

# One-Hot Encode the categorical 'terrain_type' column
print("Encoding categorical features...")
df = pd.get_dummies(df, columns=["terrain_type"], drop_first=False)

# Update features list with the new dummy columns
terrain_cols = [col for col in df.columns if col.startswith("terrain_type_")]
X_columns = features + terrain_cols

X = df[X_columns]

# Define our 4 distinct targets
y_crime = df['Crime_Risk']
y_accident = df['Accident_Risk']
y_env = df['Environment_Risk']
y_iso = df['Isolation_Risk']

# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================
print("Splitting data (80% Train / 20% Test)...")
X_train, X_test, y_crime_train, y_crime_test = train_test_split(X, y_crime, test_size=0.2, random_state=42)
_, _, y_acc_train, y_acc_test = train_test_split(X, y_accident, test_size=0.2, random_state=42)
_, _, y_env_train, y_env_test = train_test_split(X, y_env, test_size=0.2, random_state=42)
_, _, y_iso_train, y_iso_test = train_test_split(X, y_iso, test_size=0.2, random_state=42)

# ==========================================
# 3. CONFIGURE XGBOOST
# ==========================================
# Using hist method for massive speedups on large datasets
xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'tree_method': 'hist', 
    'random_state': 42,
    'n_jobs': -1 # Use all CPU cores
}

def train_and_evaluate(target_name, y_train, y_test):
    print(f"\n--- Training {target_name} Model ---")
    model = xgb.XGBRegressor(**xgb_params)
    
    train_start = time.time()
    model.fit(X_train, y_train)
    print(f"Training took {time.time() - train_start:.2f} seconds.")
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"[Results for {target_name}]")
    print(f"MAE  (Mean Absolute Error): {mae:.2f} points")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} points")
    print(f"R² Score: {r2:.4f}")
    
    return model

# ==========================================
# 4. TRAIN MODELS
# ==========================================
model_crime = train_and_evaluate("Crime Risk", y_crime_train, y_crime_test)
model_accident = train_and_evaluate("Accident Risk", y_acc_train, y_acc_test)
model_environment = train_and_evaluate("Environment Risk", y_env_train, y_env_test)
model_isolation = train_and_evaluate("Isolation Risk", y_iso_train, y_iso_test)

# ==========================================
# 5. EXPORT MODELS
# ==========================================
print("\nExporting trained models to disk...")
os.makedirs("models", exist_ok=True)
joblib.dump(model_crime, "models/xgb_crime.pkl")
joblib.dump(model_accident, "models/xgb_accident.pkl")
joblib.dump(model_environment, "models/xgb_environment.pkl")
joblib.dump(model_isolation, "models/xgb_isolation.pkl")

# Save the expected feature column names for the future API
joblib.dump(X_columns, "models/feature_columns.pkl")

print("Success! All models trained and saved to the 'models/' directory.")
