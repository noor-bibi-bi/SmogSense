"""
Day 2b: Winter-specific validation.

Our previous test set accidentally landed in spring/summer -- the calmer
part of the year. Since SmogSense is specifically about winter smog
forecasting, we need to know how the model performs THEN, not just on
an arbitrary recent slice of days.

APPROACH:
- Train on everything BEFORE Dec 1, 2025
- Test specifically on Dec 1, 2025 -- Feb 28, 2026 (a full recent winter)
- This tells us the number that actually matters for your pitch

HOW TO RUN:
1. Make sure smogsense_training_data.csv is in the same folder
2. python winter_validation.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------- Load and engineer features (same as before) ----------
df = pd.read_csv("smogsense_training_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3_avg"] = df["pm25"].shift(1).rolling(window=3).mean()
df["pm25_lag7_avg"] = df["pm25"].shift(1).rolling(window=7).mean()
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["pm25_target"] = df["pm25"].shift(-1)

df_model = df.dropna(subset=[
    "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg", "pm25_target"
]).reset_index(drop=True)

feature_cols = [
    "pm25", "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg",
    "temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm",
    "month", "day_of_year",
]

# ---------- Winter-specific split ----------
WINTER_TEST_START = "2025-12-01"
WINTER_TEST_END = "2026-02-28"

train_mask = df_model["date"] < WINTER_TEST_START
test_mask = (df_model["date"] >= WINTER_TEST_START) & (df_model["date"] <= WINTER_TEST_END)

X_train, y_train = df_model.loc[train_mask, feature_cols], df_model.loc[train_mask, "pm25_target"]
X_test, y_test = df_model.loc[test_mask, feature_cols], df_model.loc[test_mask, "pm25_target"]

print(f"Train set: {len(X_train)} days (everything before {WINTER_TEST_START})")
print(f"Winter test set: {len(X_test)} days ({WINTER_TEST_START} to {WINTER_TEST_END})")

if len(X_test) == 0:
    print("\nWARNING: No test data found in this winter window.")
    print("Check that your CSV actually covers this date range.")
else:
    print(f"\nActual PM2.5 range in this winter test period: {y_test.min():.0f} to {y_test.max():.0f}")

    # ---------- Train and evaluate ----------
    print("\n--- XGBoost, tested on winter smog season ---")
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"MAE:  {xgb_mae:.2f}")
    print(f"RMSE: {xgb_rmse:.2f}")

    # Naive baseline for the same winter period
    naive_preds = X_test["pm25"]
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    print("\n--- Naive baseline (\"tomorrow = today\"), same winter period ---")
    print(f"MAE:  {naive_mae:.2f}")
    print(f"RMSE: {naive_rmse:.2f}")

    improvement_pct = (naive_mae - xgb_mae) / naive_mae * 100
    print(f"\nModel improves on naive baseline by {improvement_pct:.1f}% during winter smog season")

    # Save this winter-validated model -- this is the one we'll actually use
    import joblib
    joblib.dump(xgb, "smogsense_model.pkl")
    print("\nWinter-validated model saved to smogsense_model.pkl")
