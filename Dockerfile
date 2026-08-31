FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by XGBoost and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and trained model artifacts
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run FastAPI with Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
