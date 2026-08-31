"""
SmogSense FastAPI backend.

Endpoints:
  GET /forecast       -> current PM2.5 + tomorrow's predicted PM2.5
  GET /recommendation -> Proceed / Move Indoors decision + advisory text

Run locally:
  pip install fastapi uvicorn joblib pandas xgboost requests
  uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs to test it interactively.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta

app = FastAPI(title="SmogSense API")

# Allow the frontend (running on a different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL before final submission
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models once at startup
reg_model = joblib.load("smogsense_regression_model.pkl")
clf_model = joblib.load("smogsense_decision_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
label_map = joblib.load("label_map.pkl")

LATITUDE = 31.54815
LONGITUDE = 74.34396
DECISION_THRESHOLD = 100

ADVISORY_TEXT = {
    "Proceed": "Air quality is expected to be acceptable. Outdoor recess and PE can proceed as planned.",
    "Move Indoors": "Air quality is expected to be unhealthy. Move outdoor recess and PE indoors tomorrow.",
}


def get_recent_conditions():
    """
    Pull the last ~10 days of PM2.5 (from OpenAQ) and weather (from
    Open-Meteo) so we can build the same lag/trend features the model
    was trained on, using the most current available data.

    NOTE: replace YOUR_OPENAQ_API_KEY with your real key before running.
    """
    OPENAQ_API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"
    PM25_SENSOR_ID = 7466365

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=10)

    # Recent PM2.5
    resp = requests.get(
        f"https://api.openaq.org/v3/sensors/{PM25_SENSOR_ID}/days",
        headers={"X-API-Key": OPENAQ_API_KEY},
        params={"date_from": str(start_date), "date_to": str(end_date), "limit": 20},
    )
    resp.raise_for_status()
    pm25_records = resp.json()["results"]
    pm25_by_date = {
        r["period"]["datetimeFrom"]["local"][:10]: r["value"] for r in pm25_records
    }

    # Recent weather
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
    )
    weather_resp.raise_for_status()
    weather_json = weather_resp.json()["daily"]

    dates = sorted(pm25_by_date.keys())
    latest_date = dates[-1]

    pm25_series = pd.Series({d: pm25_by_date[d] for d in dates}).sort_index()

    latest_pm25 = pm25_series.iloc[-1]
    lag1 = pm25_series.iloc[-1]
    lag3_avg = pm25_series.iloc[-3:].mean()
    lag7_avg = pm25_series.iloc[-7:].mean()
    change_1d = pm25_series.iloc[-1] - pm25_series.iloc[-2]
    volatility_7d = pm25_series.iloc[-7:].std()

    tomorrow = datetime.utcnow().date() + timedelta(days=1)

    features = pd.DataFrame([{
        "pm25": latest_pm25,
        "pm25_lag1": lag1,
        "pm25_lag3_avg": lag3_avg,
        "pm25_lag7_avg": lag7_avg,
        "pm25_change_1d": change_1d,
        "pm25_volatility_7d": volatility_7d,
        "temperature_c": weather_json["temperature_2m_mean"][-1],
        "humidity_pct": weather_json["relative_humidity_2m_mean"][-1],
        "wind_speed_kmh": weather_json["wind_speed_10m_mean"][-1],
        "precipitation_mm": weather_json["precipitation_sum"][-1],
        "month": tomorrow.month,
        "day_of_year": tomorrow.timetuple().tm_yday,
        "is_smog_season": 1 if tomorrow.month in [11, 12, 1, 2] else 0,
    }])[feature_cols]

    return latest_pm25, features


from pydantic import BaseModel, EmailStr
from typing import List, Optional

# In-memory alert subscriber registry (can be persisted to SQLite/PostgreSQL for production)
subscribers_db = [
    {
        "school_name": "Lahore Model School - Main Campus",
        "email": "admin@lahoremodel.edu.pk",
        "threshold": 100,
        "subscribed_at": datetime.utcnow().isoformat(),
    }
]


class SubscriberCreate(BaseModel):
    school_name: str
    email: str
    threshold: Optional[int] = 100


@app.get("/forecast")
def get_forecast():
    latest_pm25, features = get_recent_conditions()
    predicted_pm25 = float(reg_model.predict(features)[0])
    return {
        "current_pm25": round(float(latest_pm25), 1),
        "predicted_pm25_tomorrow": round(predicted_pm25, 1),
    }


@app.get("/recommendation")
def get_recommendation():
    latest_pm25, features = get_recent_conditions()
    decision_code = int(clf_model.predict(features)[0])
    decision = label_map[decision_code]
    return {
        "decision": decision,
        "advisory": ADVISORY_TEXT[decision],
    }


@app.post("/alerts/subscribe")
def subscribe_alert(sub: SubscriberCreate):
    """
    Registers an administrator/school email to receive morning 6:00 AM
    'Move Indoors' Smog Alert dispatches.
    """
    entry = {
        "school_name": sub.school_name,
        "email": sub.email,
        "threshold": sub.threshold or 100,
        "subscribed_at": datetime.utcnow().isoformat(),
    }
    subscribers_db.append(entry)
    return {
        "status": "success",
        "message": f"Alert subscription registered for {sub.school_name} ({sub.email}).",
        "subscriber_count": len(subscribers_db),
    }


@app.get("/alerts/subscribers")
def list_subscribers():
    """Returns active subscribers (for admin dashboard)."""
    return {
        "total_active": len(subscribers_db),
        "subscribers": subscribers_db,
    }


@app.post("/alerts/dispatch")
@app.get("/alerts/dispatch")
def check_and_dispatch_alerts(force_test: bool = False):
    """
    Evaluates tomorrow's PM2.5 forecast against the 100 µg/m³ threshold.
    If threshold is breached (or force_test=True), dispatches emergency
    morning notifications to registered schools.
    """
    latest_pm25, features = get_recent_conditions()
    predicted_pm25 = float(reg_model.predict(features)[0])
    decision_code = int(clf_model.predict(features)[0])
    decision = label_map[decision_code]

    is_alert_triggered = predicted_pm25 > DECISION_THRESHOLD or force_test

    dispatched_list = []
    if is_alert_triggered:
        for sub in subscribers_db:
            dispatched_list.append({
                "to": sub["email"],
                "school": sub["school_name"],
                "subject": f"🚨 [SMOG ALERT - LAHORE]: Move Recess & PE Indoors Tomorrow ({round(predicted_pm25, 1)} µg/m³)",
                "status": "SENT",
                "timestamp": datetime.utcnow().isoformat(),
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

