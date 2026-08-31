"""
Step 3c: Pull historical weather data from Open-Meteo for the same
location and date range as our PM2.5 data, then merge everything into
one final training dataset.

Open-Meteo needs NO API key -- it's fully open.

HOW TO RUN THIS (on your own machine):
1. pip install requests (if not already installed)
2. python pull_weather_and_merge.py
3. This creates "smogsense_training_data.csv" -- our final merged
   dataset, ready for Day 2 (feature engineering + model training)
"""

import requests
import csv

# Same coordinates as our chosen OpenAQ "Lahore" station
LATITUDE = 31.54815
LONGITUDE = 74.34396

DATE_FROM = "2023-11-28"
DATE_TO = "2026-08-20"  # yesterday, since today may be incomplete

print("Fetching historical weather data from Open-Meteo...")

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": DATE_FROM,
    "end_date": DATE_TO,
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "precipitation_sum",
    ],
    "timezone": "Asia/Karachi",
}

response = requests.get(url, params=params)
response.raise_for_status()
weather_data = response.json()

daily = weather_data["daily"]
weather_by_date = {}
for i, date in enumerate(daily["time"]):
    weather_by_date[date] = {
        "temperature_c": daily["temperature_2m_mean"][i],
        "humidity_pct": daily["relative_humidity_2m_mean"][i],
        "wind_speed_kmh": daily["wind_speed_10m_mean"][i],
        "precipitation_mm": daily["precipitation_sum"][i],
    }

print(f"Fetched weather for {len(weather_by_date)} days")

# Now read our existing PM2.5 CSV and merge
print("\nMerging with PM2.5 data...")

merged_rows = []
with open("lahore_historical_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        date = row["date"]
        pm25 = row["pm25"]

        weather = weather_by_date.get(date, {})

        merged_rows.append({
            "date": date,
            "pm25": pm25,
            "temperature_c": weather.get("temperature_c", ""),
            "humidity_pct": weather.get("humidity_pct", ""),
            "wind_speed_kmh": weather.get("wind_speed_kmh", ""),
            "precipitation_mm": weather.get("precipitation_mm", ""),
        })

with open("smogsense_training_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "date", "pm25", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "precipitation_mm"
    ])
    writer.writeheader()
    writer.writerows(merged_rows)

# Quick data quality check
missing_pm25 = sum(1 for r in merged_rows if not r["pm25"])
missing_weather = sum(1 for r in merged_rows if not r["temperature_c"])

print(f"\nDone! Saved {len(merged_rows)} merged rows to smogsense_training_data.csv")
print(f"Rows missing PM2.5: {missing_pm25}")
print(f"Rows missing weather: {missing_weather}")
