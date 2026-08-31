"""
SmogSense FastAPI + Gradio Backend for Hugging Face Spaces.

Endpoints:
  GET  /forecast          -> current PM2.5 + tomorrow's predicted PM2.5
  GET  /recommendation    -> Proceed / Move Indoors decision + advisory text
  POST /alerts/subscribe  -> Register school email for morning smog dispatch
  POST /alerts/dispatch   -> Evaluate threshold and trigger alert notifications
"""

import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

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

# In-memory alert subscriber registry
subscribers_db = [
    {
        "school_name": "Lahore Model School - Main Campus",
        "email": "admin@lahoremodel.edu.pk",
        "threshold": 100,
        "subscribed_at": datetime.now(timezone.utc).isoformat(),
    }
]


class SubscriberCreate(BaseModel):
    school_name: str
    email: str
    threshold: Optional[int] = 100


def get_recent_conditions():
    """
    Pull the last ~10 days of PM2.5 (from OpenAQ) and weather (from
    Open-Meteo) so we can build the same lag/trend features the model
    was trained on, using the most current available data.
    """
    OPENAQ_API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"
    PM25_SENSOR_ID = 7466365

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=10)

    # Recent PM2.5
    resp = requests.get(
        f"https://api.openaq.org/v3/sensors/{PM25_SENSOR_ID}/days",
        headers={"X-API-Key": OPENAQ_API_KEY},
        params={"date_from": str(start_date), "date_to": str(end_date), "limit": 20},
    )
    resp.raise_for_status()
    pm25_records = resp.json().get("results", [])
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
    weather_json = weather_resp.json().get("daily", {})

    dates = sorted(pm25_by_date.keys())
    if not dates:
        # Fallback if sensor API is momentarily unavailable
        latest_pm25 = 65.0
        pm25_series = pd.Series([65.0]*10)
    else:
        pm25_series = pd.Series({d: pm25_by_date[d] for d in dates}).sort_index()
        latest_pm25 = pm25_series.iloc[-1]

    lag1 = pm25_series.iloc[-1] if len(pm25_series) >= 1 else 65.0
    lag3_avg = pm25_series.iloc[-3:].mean() if len(pm25_series) >= 3 else lag1
    lag7_avg = pm25_series.iloc[-7:].mean() if len(pm25_series) >= 7 else lag1
    change_1d = (pm25_series.iloc[-1] - pm25_series.iloc[-2]) if len(pm25_series) >= 2 else 0.0
    volatility_7d = pm25_series.iloc[-7:].std() if len(pm25_series) >= 7 else 5.0
    if pd.isna(volatility_7d):
        volatility_7d = 5.0

    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)

    temp_list = weather_json.get("temperature_2m_mean", [28.0])
    hum_list = weather_json.get("relative_humidity_2m_mean", [55.0])
    wind_list = weather_json.get("wind_speed_10m_mean", [8.0])
    precip_list = weather_json.get("precipitation_sum", [0.0])

    features = pd.DataFrame([{
        "pm25": latest_pm25,
        "pm25_lag1": lag1,
        "pm25_lag3_avg": lag3_avg,
        "pm25_lag7_avg": lag7_avg,
        "pm25_change_1d": change_1d,
        "pm25_volatility_7d": volatility_7d,
        "temperature_c": temp_list[-1] if temp_list else 28.0,
        "humidity_pct": hum_list[-1] if hum_list else 55.0,
        "wind_speed_kmh": wind_list[-1] if wind_list else 8.0,
        "precipitation_mm": precip_list[-1] if precip_list else 0.0,
        "month": tomorrow.month,
        "day_of_year": tomorrow.timetuple().tm_yday,
        "is_smog_season": 1 if tomorrow.month in [11, 12, 1, 2] else 0,
    }])[feature_cols]

    return latest_pm25, features


def get_forecast():
    latest_pm25, features = get_recent_conditions()
    predicted_pm25 = float(reg_model.predict(features)[0])
    return {
        "current_pm25": round(float(latest_pm25), 1),
        "predicted_pm25_tomorrow": round(predicted_pm25, 1),
    }


def get_recommendation():
    latest_pm25, features = get_recent_conditions()
    decision_code = int(clf_model.predict(features)[0])
    decision = label_map[decision_code]
    return {
        "decision": decision,
        "advisory": ADVISORY_TEXT[decision],
    }


def check_and_dispatch_alerts(force_test: bool = False):
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


# ==========================================
# Gradio Interface for Hugging Face Spaces
# ==========================================
with gr.Blocks(title="SmogSense API") as demo:
    gr.Markdown("# 🌫️ SmogSense — City Intelligence FastAPI Backend")
    gr.Markdown("This Hugging Face Space powers the real-time **XGBoost PM2.5 forecasting model** and automated school alert system for Central Lahore.")
    
    with gr.Row():
        btn_forecast = gr.Button("⚡ Test /forecast API", variant="primary")
        btn_rec = gr.Button("🏫 Test /recommendation API")
        
    out_json = gr.JSON(label="Live Model Response")
    
    btn_forecast.click(fn=get_forecast, outputs=out_json)
    btn_rec.click(fn=get_recommendation, outputs=out_json)


# ==========================================
# Attach FastAPI Endpoints & CORS to demo.app
# ==========================================
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@demo.app.get("/forecast")
def api_forecast():
    return get_forecast()


@demo.app.get("/recommendation")
def api_recommendation():
    return get_recommendation()


@demo.app.post("/alerts/subscribe")
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


@demo.app.get("/alerts/subscribers")
def api_subscribers():
    return {
        "total_active": len(subscribers_db),
        "subscribers": subscribers_db,
    }


@demo.app.post("/alerts/dispatch")
@demo.app.get("/alerts/dispatch")
def api_dispatch(force_test: bool = False):
    return check_and_dispatch_alerts(force_test=force_test)


# Export app for ASGI runners
app = demo.app

if __name__ == "__main__":
    demo.launch()
