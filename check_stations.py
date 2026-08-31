"""
Step 2: Check which OpenAQ stations exist near Lahore, and how much
historical data each one has, before committing to a target area.

HOW TO RUN THIS (on your own machine, since Claude's sandbox can't
reach openaq.org directly):
1. pip install requests
2. Replace YOUR_OPENAQ_API_KEY below with your real key
3. python check_stations.py
"""

import requests

API_KEY = "cf98e0e95d84adce03f4bd242154623c0a3132a8ee7c4974fd43531750717f05"  # <-- put your real OpenAQ key here

# Lahore's approximate center coordinates
LAHORE_LAT = 31.5497
LAHORE_LON = 74.3436
RADIUS_METERS = 25000  # 25km radius, covers most of urban Lahore

url = "https://api.openaq.org/v3/locations"
params = {
    "coordinates": f"{LAHORE_LAT},{LAHORE_LON}",
    "radius": RADIUS_METERS,
    "limit": 100,
}
headers = {"X-API-Key": API_KEY}

response = requests.get(url, params=params, headers=headers)
response.raise_for_status()
data = response.json()

print(f"Found {len(data['results'])} station(s) near Lahore:\n")

for loc in data["results"]:
    print(f"Name: {loc['name']}")
    print(f"  ID: {loc['id']}")
    print(f"  Coordinates: {loc['coordinates']}")
    print(f"  Parameters measured: {[p['parameter']['name'] for p in loc.get('sensors', [])]}")
    print(f"  First reading: {loc.get('datetimeFirst', 'unknown')}")
    print(f"  Last reading: {loc.get('datetimeLast', 'unknown')}")
    print(f"  Is monitor active: {loc.get('isMonitor', 'unknown')}")
    print()
