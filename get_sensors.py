"""
Step 3a: Find the sensor IDs for our chosen station (ID: 1894641, "Lahore").
Each parameter (pm25, pm10, temperature, etc.) has its own sensor ID
under a location -- we need these IDs to pull actual measurements.

HOW TO RUN THIS (on your own machine):
1. Replace YOUR_OPENAQ_API_KEY below
2. python get_sensors.py
"""

import requests

API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"
LOCATION_ID = 1894641  # "Lahore" station we chose

headers = {"X-API-Key": API_KEY}

url = f"https://api.openaq.org/v3/locations/{LOCATION_ID}/sensors"
response = requests.get(url, headers=headers)
response.raise_for_status()
data = response.json()

print("Sensors at this location:\n")
for sensor in data["results"]:
    print(f"Parameter: {sensor['parameter']['name']}")
    print(f"  Sensor ID: {sensor['id']}")
    print(f"  Units: {sensor['parameter']['units']}")
    print()
