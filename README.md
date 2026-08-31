# 🌫️ SmogSense — City Intelligence for Lahore

> **AI-Powered Air Quality Forecasting & Morning School Activity Alert System**  
> *Built for the Smart City Hackathon — "City Intelligence" Theme*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EB5424?style=flat)](https://xgboost.ai)
[![OpenAQ](https://img.shields.io/badge/Data-OpenAQ%20API%20v3-0284c7?style=flat)](https://openaq.org)
[![Open-Meteo](https://img.shields.io/badge/Weather-Open--Meteo-38bdf8?style=flat)](https://open-meteo.com)
[![Theme](https://img.shields.io/badge/Smart%20City-City%20Intelligence-10b981?style=flat)](#)

---

## 📌 Executive Summary

Every winter, Lahore experiences catastrophic levels of hazardous particulate smog ($PM_{2.5} > 200\text{--}400\ \mu\text{g/m}^3$). While existing tools display raw historical air quality indices, **school administrators and parents face an actionable daily dilemma at 6:00 AM: *Should outdoor morning recess, physical education (PE), and sports proceed outside or move indoors?***

**SmogSense** transforms raw multi-source atmospheric data into an **actionable, predictive City Intelligence decision system**:
1. **Next-Day $PM_{2.5}$ Forecast**: Predicts central Lahore's next-day particulate levels using a trained XGBoost ensemble combining historical sensor lags with weather forecasts.
2. **Binary Recess Directive**: Issues an unequivocal **`Proceed`** ($\le 100\ \mu\text{g/m}^3$) or **`Move Indoors`** ($> 100\ \mu\text{g/m}^3$) recommendation with high lead time.
3. **Automated Alerting Layer**: Dispatches emergency morning notifications to subscribed schools and parents before morning buses depart.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   OpenAQ Sensor Feed   │
                                  │  (Central Lahore 7466) │
                                  └───────────┬────────────┘
                                              │ 10-day PM2.5 Lags
                                              ▼
┌────────────────────────┐        ┌────────────────────────┐        ┌────────────────────────┐
│ Open-Meteo Weather API │ ─────► │  Feature Engineering   │ ─────► │ XGBoost Dual Engine    │
│ (Temp, Wind, Humidity) │        │ (Rolling 1d,3d,7d,Vol) │        │ (Regressor+Classifier) │
└────────────────────────┘        └────────────────────────┘        └───────────┬────────────┘
                                                                                │
                                                                                ▼
┌────────────────────────┐        ┌────────────────────────┐        ┌────────────────────────┐
│ Institutional Alerting │ ◄───── │ FastAPI Microservice   │ ◄───── │ Single-Page Web App    │
│ (Web Push & Emails)    │        │ (/forecast, /alerts)   │        │ (Dark Civic Dashboard) │
└────────────────────────┘        └────────────────────────┘        └────────────────────────┘
```

---

## 🔬 Core Insights & Engineering Breakthroughs

### 1. The Winter-Validation Discovery
Initial baseline models trained uniformly across summer and monsoon months failed during seasonal temperature inversions. We created dedicated out-of-time winter validation windows (`2025-12-01` to `2026-02-28`) to properly evaluate model generalization during peak smog season.

### 2. Bimodal Air Quality Distribution (Safe vs. Severe)
Exploratory data analysis of Lahore's atmosphere revealed that during winter, intermediate "Moderate" days are statistically rare — conditions bifurcate sharply into clear weather or severe inversions. Rather than relying solely on continuous regression, we built a **dual-model architecture**:
- **XGBoost Classifier**: Provides a highly calibrated binary decision (`Proceed` vs `Move Indoors`) optimized against public health risk.
- **XGBoost Regressor**: Generates continuous values powering the 7-day trend chart and forecast trajectory.

---

## ⚡ API Contract Reference

The FastAPI backend exposes the following clean endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/forecast` | Returns `{ current_pm25, predicted_pm25_tomorrow }` |
| `GET` | `/recommendation` | Returns `{ decision: "Proceed" \| "Move Indoors", advisory: "..." }` |
| `POST`| `/alerts/subscribe` | Registers school email and threshold for automated dispatch |
| `POST`| `/alerts/dispatch` | Evaluates threshold & dispatches alerts (with `?force_test=true` support) |
| `GET` | `/alerts/subscribers` | Lists active school subscriptions |

---

## 💻 Quick Start & Local Run

### Prerequisites
- Python 3.10+
- Modern Web Browser

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/SmogSense.git
cd SmogSense
pip install -r requirements.txt
```

### 2. Launch the FastAPI Backend
```bash
uvicorn app:app --reload --port 8000
```
*API interactive documentation will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

### 3. Open the Frontend
Simply open `index.html` in your browser (or use VS Code Live Server / Python HTTP server):
```bash
python -m http.server 3000
```
Navigate to `http://localhost:3000`.

---

## 🌐 Production Deployment

- **Backend**: Ready for 1-click deployment on [Render](https://render.com) (`render.yaml`) or Hugging Face Spaces.
- **Frontend**: Deploy `index.html` to [Vercel](https://vercel.com), [Netlify](https://netlify.com), or GitHub Pages. Update `API_BASE` in `index.html` to point to your live backend domain.

---

## 👥 Hackathon Team & Acknowledgements

- **Theme**: City Intelligence — Smart City Hackathon
- **Data Providers**: OpenAQ Community Sensor Network & Open-Meteo Historical Weather API.
