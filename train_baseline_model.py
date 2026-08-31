"""
Day 2: Feature engineering + baseline model for next-day PM2.5 forecasting.

WHAT THIS SCRIPT DOES:
1. Loads your merged dataset
2. Creates lagged features (yesterday's PM2.5, 3-day avg, 7-day avg)
3. Creates time features (month, day of year) to capture seasonality
4. Splits data chronologically into train/test (not random -- this matters
   for time series, since we want to simulate "predicting the future"
   using only past data)
5. Trains a baseline model and reports how good it is

HOW TO RUN THIS:
1. pip install pandas scikit-learn xgboost
2. Make sure smogsense_training_data.csv is in the same folder
3. python train_baseline_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------- 1. Load data ----------
df = pd.read_csv("smogsense_training_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} rows, from {df['date'].min().date()} to {df['date'].max().date()}")

# ---------- 2. Feature engineering ----------

# Lagged PM2.5 features -- yesterday's value is usually the single
# strongest predictor of tomorrow's value in air quality data
df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3_avg"] = df["pm25"].shift(1).rolling(window=3).mean()
df["pm25_lag7_avg"] = df["pm25"].shift(1).rolling(window=7).mean()

# Time features -- captures the seasonal smog pattern you already confirmed
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# Our TARGET: tomorrow's PM2.5 (shift -1 means "the next row's value")
df["pm25_target"] = df["pm25"].shift(-1)

# Drop rows with missing values created by lagging/shifting
# (first 7 rows and last row will have NaNs -- this is expected and fine)
df_model = df.dropna(subset=[
    "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg", "pm25_target"
]).reset_index(drop=True)

print(f"After creating features: {len(df_model)} usable rows")

feature_cols = [
    "pm25", "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg",
    "temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm",
    "month", "day_of_year",
]

X = df_model[feature_cols]
y = df_model["pm25_target"]

# ---------- 3. Chronological train/test split ----------
# We train on the OLDER 85% of days and test on the MOST RECENT 15%.
# This simulates real forecasting: predicting days you haven't seen yet,
# using only what happened before them.
split_idx = int(len(df_model) * 0.85)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain set: {len(X_train)} days ({df_model['date'].iloc[0].date()} to {df_model['date'].iloc[split_idx-1].date()})")
print(f"Test set: {len(X_test)} days ({df_model['date'].iloc[split_idx].date()} to {df_model['date'].iloc[-1].date()})")

# ---------- 4. Train baseline models ----------

print("\n--- Linear Regression (simplest baseline) ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
print(f"MAE:  {lr_mae:.2f} (avg. AQI points off, on average)")
print(f"RMSE: {lr_rmse:.2f}")

print("\n--- XGBoost (stronger model) ---")
xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"MAE:  {xgb_mae:.2f}")
print(f"RMSE: {xgb_rmse:.2f}")

# ---------- 5. Also compare against a "naive" baseline ----------
# The simplest possible forecast: "tomorrow will be the same as today"
# If our model can't beat this, it's not actually learning anything useful.
naive_preds = X_test["pm25"]  # today's value, used as "prediction" for tomorrow
naive_mae = mean_absolute_error(y_test, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
print("\n--- Naive baseline (\"tomorrow = today\") for comparison ---")
print(f"MAE:  {naive_mae:.2f}")
print(f"RMSE: {naive_rmse:.2f}")

# ---------- 6. Feature importance (XGBoost) ----------
print("\n--- Which features matter most? (XGBoost) ---")
importances = sorted(
    zip(feature_cols, xgb.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for name, score in importances:
    print(f"  {name}: {score:.3f}")

# Save the trained model for later use in the FastAPI backend
import joblib
joblib.dump(xgb, "smogsense_model.pkl")
print("\nModel saved to smogsense_model.pkl")
