# CodeRed Spatial Geofencing AI

This repository contains the standalone Python Spatial Engine for the **CodeRed** platform. It acts as an independent Microservice that takes live GPS coordinates, mathematically generates risk features, predicts dynamic danger levels using XGBoost ML models, and manages Shapely Geofencing bounds and Alert Hysteresis.

## 🏗️ 3-Layer Architecture

1.  **Layer 1: ML Prediction Engine**
    *   4 independent XGBoost Regressors predicting Crime, Accident, Environment, and Isolation risk scores (0-100).
    *   Models were trained on a highly detailed 3.5-million row synthetic dataset simulating Kanpur city patterns over 14 days.
2.  **Layer 2: Spatial Geofencing**
    *   Divides Kanpur into 10,453 unique **H3 Hexagons** (Resolution 9).
    *   When multiple hexagons trigger a `CRITICAL` state (score >= 70), they are melted together using `Shapely` to form a unified GeoJSON polygon, and expanded with a **200m Warning Buffer**.
3.  **Layer 3: Hysteresis & API**
    *   Packaged as a lightning-fast **FastAPI** microservice.
    *   Manages a 5-minute Alert Cooldown (Hysteresis) per user to prevent notification spam if a GPS signal bounces across a border.

---

## 🚀 Getting Started

You can run this project either natively using Python or flawlessly via Docker.

### Option A: Run via Docker (Recommended for Node.js/Frontend teams)
If you don't want to install Python dependencies, simply use Docker Compose:
```bash
docker-compose up -d --build
```
The FastAPI server is now running natively on port `8000`.

### Option B: Run via Python
If you want to edit the ML code or re-train models, use Python directly:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Data Simulation (Optional)
python src/generate_data.py

# 3. Train the XGBoost Models (Optional)
python src/train_models.py

# 4. Start the Inference API Server
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📡 Using the API

The CodeRed Node.js backend should query this Python Microservice whenever a tourist's app pings a GPS location.

### `POST /api/v1/geofence/check`

**Request Payload:**
```json
{
  "user_id": "tourist_1",
  "lat": 26.4950,
  "lng": 80.3150
}
```

**Response Payload:**
```json
{
  "user_id": "tourist_1",
  "overall_score": 41.2,
  "status": "CAUTION",
  "dispatch_alert": false,
  "details": {
    "Crime_Risk": 33.07,
    "Accident_Risk": 18.93,
    "Environment_Risk": 99.94,
    "Isolation_Risk": 24.53
  }
}
```

*Note: You only need to send an SOS/Notification to the user if `"dispatch_alert": true`. The Python AI automatically handles the 5-minute cooldown logic for you.*

### 🗺️ Live Map Prototype
To see the Geofences rendered visually, run:
```bash
python src/predict.py
```
This will scan all 10,000+ hexagons in Kanpur, merge critical zones using Shapely, and generate an interactive `live_geofence_map.html` in the root folder.
