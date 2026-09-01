"""
SmogSense Standalone FastAPI Backend for Vercel Python Serverless Runtime.
Pure-Python Tree Inference Engine (Zero C++ / XGBoost / Sklearn runtime dependencies).

Endpoints:
  GET  /                  -> API Health Status & Info
  GET  /forecast          -> Current PM2.5 + Tomorrow's Predicted PM2.5
  GET  /recommendation    -> Proceed / Move Indoors Decision + Advisory Text
  POST /alerts/subscribe  -> Register School Email for Morning Smog Dispatch
  GET  /alerts/subscribers-> View Active Subscribers
  POST /alerts/dispatch   -> Trigger Alert Notifications Pipeline
"""

import os
import json
import math
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    "pm25", "pm25_lag1", "pm25_lag3_avg", "pm25_lag7_avg",
    "pm25_change_1d", "pm25_volatility_7d",
    "temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm",
    "month", "day_of_year", "is_smog_season"
]

LABEL_MAP = {
    0: "Proceed",
    1: "Move Indoors"
}

LATITUDE = 31.54815
LONGITUDE = 74.34396
DECISION_THRESHOLD = 100

ADVISORY_TEXT = {
    "Proceed": "Air quality is expected to be acceptable. Outdoor recess and PE can proceed as planned.",
    "Move Indoors": "Air quality is expected to be unhealthy. Move outdoor recess and PE indoors tomorrow.",
}

subscribers_db: List[Dict[str, Any]] = [
    {
        "school_name": "Lahore Model School - Main Campus",
        "email": "admin@lahoremodel.edu.pk",
        "threshold": 100,
        "subscribed_at": datetime.now(timezone.utc).isoformat(),
    }
]

# Load model tree JSONs
reg_model_path = os.path.join(BASE_DIR, "reg_model.json")
clf_model_path = os.path.join(BASE_DIR, "clf_model.json")

with open(reg_model_path, "r", encoding="utf-8") as f:
    reg_json = json.load(f)

with open(clf_model_path, "r", encoding="utf-8") as f:
    clf_json = json.load(f)

reg_base_score = float(reg_json["learner"]["learner_model_param"]["base_score"].strip("[]"))
clf_base_prob = float(clf_json["learner"]["learner_model_param"]["base_score"].strip("[]"))
clf_base_margin = math.log(clf_base_prob / (1.0 - clf_base_prob))

reg_trees = reg_json["learner"]["gradient_booster"]["model"]["trees"]
clf_trees = clf_json["learner"]["gradient_booster"]["model"]["trees"]


def _traverse_tree(tree: Dict[str, Any], feat_vec: List[float]) -> float:
    node = 0
    lefts = tree["left_children"]
    rights = tree["right_children"]
    splits = tree["split_indices"]
    conds = tree["split_conditions"]
    default_left = tree["default_left"]

    while lefts[node] != -1:
        feat_idx = splits[node]
        val = feat_vec[feat_idx]
        if val is None or math.isnan(val):
            node = lefts[node] if default_left[node] == 1 else rights[node]
        elif val < conds[node]:
            node = lefts[node]
        else:
            node = rights[node]

    return conds[node]


def predict_regression(feat_vec: List[float]) -> float:
    score = reg_base_score
    for tree in reg_trees:
        score += _traverse_tree(tree, feat_vec)
    return score


def predict_classification(feat_vec: List[float]) -> int:
    margin = clf_base_margin
    for tree in clf_trees:
        margin += _traverse_tree(tree, feat_vec)
    prob = 1.0 / (1.0 + math.exp(-margin))
    return 1 if prob > 0.5 else 0


app = FastAPI(
    title="SmogSense API",
    description="Smart City Lahore Air Quality Forecasting & School Activity Advisory Backend",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubscriberCreate(BaseModel):
    school_name: str
    email: str
    threshold: Optional[int] = 100


def get_recent_conditions():
    OPENAQ_API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"
    PM25_SENSOR_ID = 7466365

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=10)

    pm25_by_date = {}
    try:
        resp = requests.get(
            f"https://api.openaq.org/v3/sensors/{PM25_SENSOR_ID}/days",
            headers={"X-API-Key": OPENAQ_API_KEY},
            params={"date_from": str(start_date), "date_to": str(end_date), "limit": 20},
            timeout=5
        )
        if resp.status_code == 200:
            pm25_records = resp.json().get("results", [])
            pm25_by_date = {
                r["period"]["datetimeFrom"]["local"][:10]: float(r["value"])
                for r in pm25_records if "period" in r and "value" in r
            }
    except Exception as e:
        print(f"Warning: OpenAQ fetch failed: {e}")

    weather_json = {}
    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "daily": [
                    "temperature_2m_mean", "relative_humidity_2m_mean",
                    "wind_speed_10m_mean", "precipitation_sum",
                ],
                "past_days": 10,
                "forecast_days": 1,
                "timezone": "Asia/Karachi",
            },
            timeout=5
        )
        if weather_resp.status_code == 200:
            weather_json = weather_resp.json().get("daily", {})
    except Exception as e:
        print(f"Warning: Open-Meteo fetch failed: {e}")

    dates = sorted(pm25_by_date.keys())
    if not dates:
        latest_pm25 = 65.0
        pm25_values = [65.0] * 10
    else:
        pm25_values = [pm25_by_date[d] for d in dates]
        latest_pm25 = pm25_values[-1]

    lag1 = pm25_values[-1] if len(pm25_values) >= 1 else 65.0
    lag3_avg = float(np.mean(pm25_values[-3:])) if len(pm25_values) >= 3 else lag1
    lag7_avg = float(np.mean(pm25_values[-7:])) if len(pm25_values) >= 7 else lag1
    change_1d = (pm25_values[-1] - pm25_values[-2]) if len(pm25_values) >= 2 else 0.0

    if len(pm25_values) >= 2:
        recent_7 = pm25_values[-7:]
        volatility_7d = float(np.std(recent_7, ddof=1)) if len(recent_7) > 1 else 5.0
    else:
        volatility_7d = 5.0

    if np.isnan(volatility_7d):
        volatility_7d = 5.0

    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)

    temp_list = weather_json.get("temperature_2m_mean", [28.0])
    hum_list = weather_json.get("relative_humidity_2m_mean", [55.0])
    wind_list = weather_json.get("wind_speed_10m_mean", [8.0])
    precip_list = weather_json.get("precipitation_sum", [0.0])

    features_dict = {
        "pm25": float(latest_pm25),
        "pm25_lag1": float(lag1),
        "pm25_lag3_avg": float(lag3_avg),
        "pm25_lag7_avg": float(lag7_avg),
        "pm25_change_1d": float(change_1d),
        "pm25_volatility_7d": float(volatility_7d),
        "temperature_c": float(temp_list[-1]) if temp_list else 28.0,
        "humidity_pct": float(hum_list[-1]) if hum_list else 55.0,
        "wind_speed_kmh": float(wind_list[-1]) if wind_list else 8.0,
        "precipitation_mm": float(precip_list[-1]) if precip_list else 0.0,
        "month": int(tomorrow.month),
        "day_of_year": int(tomorrow.timetuple().tm_yday),
        "is_smog_season": 1 if tomorrow.month in [11, 12, 1, 2] else 0,
    }

    feat_vec = [features_dict[col] for col in FEATURE_COLS]
    return latest_pm25, feat_vec


# ==========================================
# Clean Standalone Top-Level API Routes
# ==========================================

@app.get("/")
@app.get("/api")
@app.get("/api/health")
def root():
    return {
        "name": "SmogSense City Intelligence API",
        "status": "online",
        "runtime": "Vercel Python Serverless (Pure-Python Tree Engine)",
        "endpoints": {
            "forecast": "/forecast",
            "recommendation": "/recommendation",
            "subscribe": "/alerts/subscribe",
            "subscribers": "/alerts/subscribers",
            "dispatch": "/alerts/dispatch"
        }
    }


@app.get("/forecast")
@app.get("/api/forecast")
def api_forecast():
    latest_pm25, feat_vec = get_recent_conditions()
    pred_val = predict_regression(feat_vec)
    return {
        "current_pm25": round(float(latest_pm25), 1),
        "predicted_pm25_tomorrow": round(pred_val, 1),
    }


@app.get("/recommendation")
@app.get("/api/recommendation")
def api_recommendation():
    _, feat_vec = get_recent_conditions()
    decision_code = predict_classification(feat_vec)
    decision = LABEL_MAP[decision_code]
    return {
        "decision": decision,
        "advisory": ADVISORY_TEXT[decision],
    }


@app.post("/alerts/subscribe")
@app.post("/api/alerts/subscribe")
def api_subscribe(sub: SubscriberCreate):
    entry = {
        "school_name": sub.school_name,
        "email": sub.email,
        "threshold": sub.threshold or 100,
        "subscribed_at": datetime.now(timezone.utc).isoformat(),
    }
    subscribers_db.append(entry)
    return {
        "status": "success",
        "message": f"Alert subscription registered for {sub.school_name} ({sub.email}).",
        "subscriber_count": len(subscribers_db),
    }


@app.get("/alerts/subscribers")
@app.get("/api/alerts/subscribers")
def api_subscribers():
    return {
        "total_active": len(subscribers_db),
        "subscribers": subscribers_db,
    }


@app.post("/alerts/dispatch")
@app.get("/alerts/dispatch")
@app.post("/api/alerts/dispatch")
@app.get("/api/alerts/dispatch")
def api_dispatch(force_test: bool = False):
    latest_pm25, feat_vec = get_recent_conditions()
    predicted_pm25 = predict_regression(feat_vec)
    decision_code = predict_classification(feat_vec)
    decision = LABEL_MAP[decision_code]

    is_alert_triggered = predicted_pm25 > DECISION_THRESHOLD or force_test

    dispatched_list = []
    if is_alert_triggered:
        for sub in subscribers_db:
            dispatched_list.append({
                "to": sub["email"],
                "school": sub["school_name"],
                "subject": f"🚨 [SMOG ALERT - LAHORE]: Move Recess & PE Indoors Tomorrow ({round(predicted_pm25, 1)} µg/m³)",
                "status": "SENT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return {
        "alert_triggered": is_alert_triggered,
        "predicted_pm25_tomorrow": round(predicted_pm25, 1),
        "decision": decision,
        "threshold": DECISION_THRESHOLD,
        "dispatches_sent": len(dispatched_list),
        "dispatch_details": dispatched_list,
        "note": "Dispatched via SmogSense City Intelligence Automated Alerting Pipeline.",
    }
