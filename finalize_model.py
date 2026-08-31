"""
FINAL model: binary risk classification (Proceed / Move Indoors),
based on what the winter data actually showed (near-empty "Moderate"
category). This becomes the model your FastAPI backend serves.

Also trains the regression model alongside, for showing the forecast
TREND CHART in the UI (numbers are still useful for the chart even if
the binary decision is the headline feature).

Run: python finalize_model.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
import joblib

df = pd.read_csv("smogsense_training_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3_avg"] = df["pm25"].shift(1).rolling(window=3).mean()
df["pm25_lag7_avg"] = df["pm25"].shift(1).rolling(window=7).mean()
df["pm25_change_1d"] = df["pm25"].shift(1) - df["pm25"].shift(2)
df["pm25_volatility_7d"] = df["pm25"].shift(1).rolling(window=7).std()
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_smog_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)
df["pm25_target"] = df["pm25"].shift(-1)

# BINARY decision threshold: PM2.5 > 100 = "Move Indoors" advisory.
# (100 is a reasonable "unhealthy for sensitive groups" style cutoff --
# adjust this single number later if you want a stricter/looser rule.)
DECISION_THRESHOLD = 100

df["decision_target"] = np.where(
    df["pm25_target"] > DECISION_THRESHOLD, "Move Indoors", "Proceed"
)
df.loc[df["pm25_target"].isna(), "decision_target"] = np.nan

df_model = df.dropna(subset=[
    "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg",
    "pm25_change_1d", "pm25_volatility_7d", "pm25_target"
]).reset_index(drop=True)

feature_cols = [
    "pm25", "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg",
    "pm25_change_1d", "pm25_volatility_7d",
    "temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm",
    "month", "day_of_year", "is_smog_season",
]

WINTER_TEST_START = "2025-12-01"
WINTER_TEST_END = "2026-02-28"
train_mask = df_model["date"] < WINTER_TEST_START
test_mask = (df_model["date"] >= WINTER_TEST_START) & (df_model["date"] <= WINTER_TEST_END)

X_train = df_model.loc[train_mask, feature_cols]
X_test = df_model.loc[test_mask, feature_cols]
y_train_reg = df_model.loc[train_mask, "pm25_target"]
y_test_reg = df_model.loc[test_mask, "pm25_target"]
y_train_dec = df_model.loc[train_mask, "decision_target"]
y_test_dec = df_model.loc[test_mask, "decision_target"]

# Encode binary labels
label_map = {"Proceed": 0, "Move Indoors": 1}
inverse_label_map = {0: "Proceed", 1: "Move Indoors"}
y_train_dec_enc = y_train_dec.map(label_map)
y_test_dec_enc = y_test_dec.map(label_map)

print(f"Train: {len(X_train)} days | Winter test: {len(X_test)} days\n")

# Regression (for the trend chart)
reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
reg.fit(X_train, y_train_reg)
reg_preds = reg.predict(X_test)
print(f"Regression MAE: {mean_absolute_error(y_test_reg, reg_preds):.2f}")

# Binary classification (the actual product decision)
clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
clf.fit(X_train, y_train_dec_enc)
clf_preds_enc = clf.predict(X_test)
accuracy = accuracy_score(y_test_dec_enc, clf_preds_enc)

naive_dec = X_test["pm25"].apply(lambda x: "Move Indoors" if x > DECISION_THRESHOLD else "Proceed")
naive_dec_enc = naive_dec.map(label_map)
naive_accuracy = accuracy_score(y_test_dec_enc, naive_dec_enc)

print(f"\nDecision accuracy: {accuracy:.1%}")
print(f"Naive baseline accuracy: {naive_accuracy:.1%}")
print("\n" + classification_report(
    y_test_dec_enc, clf_preds_enc, target_names=["Proceed", "Move Indoors"], zero_division=0
))

# Save everything the backend will need
joblib.dump(reg, "smogsense_regression_model.pkl")
joblib.dump(clf, "smogsense_decision_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(inverse_label_map, "label_map.pkl")
print("Final models saved: smogsense_regression_model.pkl, smogsense_decision_model.pkl")
