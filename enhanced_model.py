"""
Day 2c: Enhanced model -- adds volatility/trend features, and tests
BOTH a regression approach (predict exact PM2.5) AND a classification
approach (predict risk category: Safe / Moderate / Unhealthy).

WHY CLASSIFICATION MATTERS HERE:
SmogSense's actual product decision is "proceed / limit / move indoors" --
a category, not an exact number. A model can be genuinely useful for
this decision even if it can't nail the exact PM2.5 value, especially
during volatile winter periods.

HOW TO RUN:
python enhanced_model.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier

# ---------- Load and engineer features ----------
df = pd.read_csv("smogsense_training_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Original lag features
df["pm25_lag1"] = df["pm25"].shift(1)
df["pm25_lag3_avg"] = df["pm25"].shift(1).rolling(window=3).mean()
df["pm25_lag7_avg"] = df["pm25"].shift(1).rolling(window=7).mean()

# NEW: trend/volatility features
df["pm25_change_1d"] = df["pm25"].shift(1) - df["pm25"].shift(2)  # yesterday's change
df["pm25_volatility_7d"] = df["pm25"].shift(1).rolling(window=7).std()  # how volatile recently

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_smog_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)  # explicit smog season flag

df["pm25_target"] = df["pm25"].shift(-1)

# Define risk categories based on PM2.5 (standard-ish breakpoints, simplified)
def categorize(pm25):
    if pm25 <= 55:
        return "Safe"
    elif pm25 <= 150:
        return "Moderate"
    else:
        return "Unhealthy"

df["risk_category_target"] = df["pm25_target"].apply(
    lambda x: categorize(x) if pd.notna(x) else np.nan
)

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

# ---------- Winter-specific split (same as before) ----------
WINTER_TEST_START = "2025-12-01"
WINTER_TEST_END = "2026-02-28"

train_mask = df_model["date"] < WINTER_TEST_START
test_mask = (df_model["date"] >= WINTER_TEST_START) & (df_model["date"] <= WINTER_TEST_END)

X_train = df_model.loc[train_mask, feature_cols]
X_test = df_model.loc[test_mask, feature_cols]
y_train_reg = df_model.loc[train_mask, "pm25_target"]
y_test_reg = df_model.loc[test_mask, "pm25_target"]
y_train_cls = df_model.loc[train_mask, "risk_category_target"]
y_test_cls = df_model.loc[test_mask, "risk_category_target"]

print(f"Train: {len(X_train)} days | Winter test: {len(X_test)} days\n")

# ---------- Enhanced regression ----------
print("=== REGRESSION (with new trend/volatility features) ===")
xgb_reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_reg.fit(X_train, y_train_reg)
preds = xgb_reg.predict(X_test)
mae = mean_absolute_error(y_test_reg, preds)
naive_mae = mean_absolute_error(y_test_reg, X_test["pm25"])
print(f"MAE: {mae:.2f} (naive baseline: {naive_mae:.2f})")
print(f"Improvement over naive: {(naive_mae - mae) / naive_mae * 100:.1f}%")

# ---------- Classification ----------
print("\n=== CLASSIFICATION (risk category: Safe / Moderate / Unhealthy) ===")
# XGBoost expects class labels to be numeric IDs, not strings like "Safe".
# Use an explicit category order so the numeric mapping is stable and meaningful.
class_order = ["Safe", "Moderate", "Unhealthy"]

y_train_cls_encoded = pd.Categorical(y_train_cls, categories=class_order, ordered=True).codes
y_test_cls_encoded = pd.Categorical(y_test_cls, categories=class_order, ordered=True).codes

xgb_cls = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_cls.fit(X_train, y_train_cls_encoded)
preds_cls_encoded = xgb_cls.predict(X_test)
preds_cls = pd.Categorical.from_codes(preds_cls_encoded, categories=class_order, ordered=True).astype(str)

y_test_cls_labels = pd.Categorical.from_codes(y_test_cls_encoded, categories=class_order, ordered=True).astype(str)
accuracy = accuracy_score(y_test_cls_labels, preds_cls)
print(f"Accuracy: {accuracy:.1%}\n")
print("Detailed breakdown:")
print(classification_report(y_test_cls_labels, preds_cls, target_names=class_order))

# Naive classification baseline: "tomorrow's category = today's category"
naive_cls_preds = X_test["pm25"].apply(categorize)
naive_accuracy = accuracy_score(y_test_cls_labels, naive_cls_preds)
print(f"Naive baseline accuracy (\"tomorrow's category = today's\"): {naive_accuracy:.1%}")

# Save both models
import joblib
joblib.dump(xgb_reg, "smogsense_regression_model.pkl")
joblib.dump(xgb_cls, "smogsense_classification_model.pkl")
print("\nBoth models saved.")
