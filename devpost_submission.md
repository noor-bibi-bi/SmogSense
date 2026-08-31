# 📝 SmogSense — Devpost Submission Package

Copy and paste these pre-formatted sections directly into your **Devpost Project Submission** fields.

---

### Project Title
`SmogSense — AI Air Quality Forecasting & School Safety Advisor for Lahore`

### Tagline
`Actionable 24-hour predictive intelligence protecting Lahore schoolchildren from toxic smog spikes.`

---

### 💡 Inspiration
Lahore is globally recognized as one of the cities most severely affected by winter smog. While several weather apps show current or retrospective AQI numbers, there is a critical missing link for decision-makers: **parents and school principals must decide at 6:00 AM every morning whether outdoor recess, PE, and sports practices are safe to conduct.** By the time morning sensors report hazardous air, children are already on school buses. We built **SmogSense** under the "City Intelligence" theme to convert raw environmental telemetry into an advance, high-lead-time safety directive specifically built for schools.

---

### ⚙️ What It Does
1. **Predicts Next-Day PM2.5**: Utilizes a machine learning pipeline that forecasts tomorrow's PM2.5 particulate concentration for Central Lahore.
2. **Issues an Unequivocal Safety Directive**: Automatically categorizes the forecast into **`Proceed Outdoors`** or **`Move Indoors`** based on a validated $100\ \mu\text{g/m}^3$ threshold.
3. **Interactive 7-Day Trend Intelligence**: Visualizes historical PM2.5 trajectory alongside the predicted point and threshold boundaries.
4. **Automated Alerting Layer**: Allows institutions and parents to subscribe to daily early-morning dispatches (Web Push notifications / Email) so coaches and transport coordinators can pivot indoors in advance.

---

### 🛠️ How We Built It
- **Data Ingestion**: Connected to the OpenAQ API v3 to pull multi-year daily aggregated PM2.5 telemetry from Central Lahore's monitoring station (Sensor 7466365) and synchronized it with Open-Meteo's meteorological reanalysis API (temperature, wind velocity, humidity, precipitation).
- **Feature Engineering**: Extracted temporal lag signatures: 1-day lag, 3-day and 7-day rolling means, 1-day delta, 7-day volatility standard deviation, day-of-year, and seasonal smog indicators.
- **Machine Learning**: Built an ensemble using **XGBoost**:
  - `XGBRegressor` for continuous trajectory forecasting.
  - `XGBClassifier` tuned specifically on binary safety threshold classification.
- **Backend**: Microservice built with **FastAPI** exposing `/forecast`, `/recommendation`, `/alerts/subscribe`, and `/alerts/dispatch`.
- **Frontend**: Responsive civic dashboard with glassmorphism UI, Chart.js trend visualization, Lucide iconography, and Native Browser Push Notification integration.

---

### 🧗 Challenges We Ran Into
1. **The Winter-Validation Surprise**: Our initial models trained on full-year data performed deceptively well on paper, but degraded during winter inversions. We resolved this by creating dedicated out-of-time winter validation splits (`2025-12-01` to `2026-02-28`), allowing us to penalize false negatives during smog peaks.
2. **Bimodal Safe/Hazardous Pattern**: In peak smog season, air quality rarely hovers in the "moderate" zone; it tends to jump sharply between clear days and hazardous spikes. Recognizing this bimodal distribution led us to pair regression with a dedicated classification model for the binary recommendation.

---

### 🏆 Accomplishments That We're Proud Of
- Delivered a **clean, institutional-grade civic UI** that government, municipal, and educational leaders can trust.
- Sub-second API inference delivering actionable predictions before school hours.
- Built a working **Alerting Layer** with Web Push notifications to transition SmogSense from a passive dashboard to an active early-warning system.

---

### 📚 What We Learned
- How atmospheric variables (specifically wind speed and seasonal temperature drops) trigger rapid particulate accumulation in the Lahore basin.
- The importance of evaluating public-health ML models on strict threshold metrics rather than simple mean squared error alone.

---

### 🚀 What's Next for SmogSense
- Expanding from a single central station to a city-wide spatial interpolation grid covering all 10+ Lahore sensor nodes.
- Direct integration with Punjab School Education Department communication portals and WhatsApp automated dispatches.
