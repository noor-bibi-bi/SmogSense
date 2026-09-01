"""
SmogSense FastAPI Backend Entrypoint.
Re-exports the lightweight serverless FastAPI app from api/index.py.
"""

from api.index import app, calculate_forecast, calculate_recommendation, execute_alert_dispatch

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
