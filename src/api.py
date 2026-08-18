from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
from src.predict import predict_risk, GeofenceHysteresisManager, get_nearby_danger_zones

# Initialize the FastAPI app
app = FastAPI(
    title="CodeRed Geofencing API",
    description="Real-time ML risk assessment and geofence triggering for tourist safety.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Hysteresis Manager
hysteresis_manager = GeofenceHysteresisManager(cooldown_minutes=5)

class LocationRequest(BaseModel):
    user_id: str
    lat: float
    lng: float

class RiskResponse(BaseModel):
    user_id: str
    lat: float
    lng: float
    timestamp: str
    overall_score: float
    status: str
    details: dict
    dispatch_alert: bool
    nearby_danger_zones: Optional[Any] = None

@app.get("/health")
def health_check():
    """Simple endpoint to verify the Python microservice is online."""
    return {"status": "online", "message": "CodeRed ML Engine is running."}

@app.post("/api/v1/geofence/check", response_model=RiskResponse)
def check_geofence(req: LocationRequest):
    try:
        now = datetime.now()
        
        # 1. Predict exact point risk
        risk_result = predict_risk(req.lat, req.lng, current_time=now)
        
        # 2. Check if we need to dispatch an SOS (Hysteresis Check)
        dispatch = hysteresis_manager.should_trigger_alert(
            user_id=req.user_id,
            status=risk_result["status"],
            current_time=now
        )
        
        # 3. Radar: Get nearby danger zones (3km radius)
        nearby_geojson = get_nearby_danger_zones(req.lat, req.lng, radius_k=15)
        
        return RiskResponse(
            user_id=req.user_id,
            lat=req.lat,
            lng=req.lng,
            timestamp=now.isoformat(),
            overall_score=risk_result["overall_score"],
            status=risk_result["status"],
            details=risk_result["details"],
            dispatch_alert=dispatch,
            nearby_danger_zones=nearby_geojson
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# If run directly (not via uvicorn CLI), start the server
if __name__ == "__main__":
    import uvicorn
    print("Starting CodeRed ML Microservice on port 8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
