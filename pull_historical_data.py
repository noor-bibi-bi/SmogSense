"""
Step 3b: Pull historical DAILY aggregated PM2.5 and temperature data
for our chosen Lahore station, and save it to a CSV file.

We use daily aggregates (not raw hourly) because:
- Our goal is next-day AQI forecasting, so daily resolution matches that
- Daily data is much smaller/faster to pull and work with
- It smooths out noisy hourly sensor readings

HOW TO RUN THIS (on your own machine):
1. Replace YOUR_OPENAQ_API_KEY below
2. python pull_historical_data.py
3. This will create a file called "lahore_historical_data.csv" in the
   same folder -- that's your training dataset for Day 2.
"""

import requests
import csv
import time

API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"
headers = {"X-API-Key": API_KEY}

PM25_SENSOR_ID = 7466365
TEMP_SENSOR_ID = 7466366

# Pull the full history available (station started Nov 2023)
DATE_FROM = "2023-11-28"
DATE_TO = "2026-08-21"  # today


def fetch_daily_data(sensor_id, date_from, date_to):
    """Fetch all daily aggregated readings for a sensor, handling pagination."""
    all_results = []
    page = 1
    limit = 1000  # max per page

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/days"
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "page": page,
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        print(f"  Fetched page {page}, {len(results)} records (total so far: {len(all_results)})")

        if len(results) < limit:
            break  # last page

        page += 1
        time.sleep(0.5)  # be polite to the API

    return all_results


print("Fetching PM2.5 daily data...")
pm25_data = fetch_daily_data(PM25_SENSOR_ID, DATE_FROM, DATE_TO)

print("\nFetching temperature daily data...")
temp_data = fetch_daily_data(TEMP_SENSOR_ID, DATE_FROM, DATE_TO)

# Build a dictionary keyed by date for easy merging
pm25_by_date = {}
for record in pm25_data:
    date = record["period"]["datetimeFrom"]["local"][:10]  # YYYY-MM-DD
    pm25_by_date[date] = record["value"]

temp_by_date = {}
for record in temp_data:
    date = record["period"]["datetimeFrom"]["local"][:10]
    temp_by_date[date] = record["value"]

# Merge and write to CSV
all_dates = sorted(set(pm25_by_date.keys()) | set(temp_by_date.keys()))

with open("lahore_historical_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "pm25", "temperature_c"])
    for date in all_dates:
        writer.writerow([
            date,
            pm25_by_date.get(date, ""),
            temp_by_date.get(date, ""),
        ])

print(f"\nDone! Saved {len(all_dates)} days of data to lahore_historical_data.csv")
print(f"PM2.5 records: {len(pm25_data)}")
print(f"Temperature records: {len(temp_data)}")
