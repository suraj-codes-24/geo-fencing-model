from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from src.predict import predict_risk, GeofenceHysteresisManager

# Initialize the FastAPI app
app = FastAPI(
    title="CodeRed Geofencing API",
    description="Python ML Microservice for predicting live spatial risk.",
    version="1.0.0"
)

# Enable CORS for the frontend tester
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with the React app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the global Hysteresis Manager (5 min cooldown)
hysteresis_manager = GeofenceHysteresisManager(cooldown_minutes=5)

# Define the expected JSON input schema
class LocationPing(BaseModel):
    user_id: str
    lat: float
    lng: float

# Define the JSON response schema (optional but good for documentation)
class RiskResponse(BaseModel):
    user_id: str
    overall_score: float
    status: str
    dispatch_alert: bool
    details: dict

@app.get("/health")
def health_check():
    """Simple endpoint to verify the Python microservice is online."""
    return {"status": "online", "message": "CodeRed ML Engine is running."}

@app.post("/api/v1/geofence/check", response_model=RiskResponse)
def check_geofence(ping: LocationPing):
    """
    Takes a GPS ping, runs the 4 XGBoost models, checks hysteresis,
    and returns whether an emergency alert should be dispatched.
    """
    try:
        # 1. Run the ML Prediction (Layer 1 & 2)
        current_time = datetime.now()
        risk_result = predict_risk(ping.lat, ping.lng, current_time)
        
        # 2. Run the Hysteresis Check (Layer 3)
        # Only dispatch if it's CRITICAL and the user isn't in a cooldown period
        dispatch = hysteresis_manager.should_trigger_alert(
            user_id=ping.user_id,
            status=risk_result["status"],
            current_time=current_time
        )
        
        # 3. Format and return response
        return {
            "user_id": ping.user_id,
            "overall_score": risk_result["overall_score"],
            "status": risk_result["status"],
            "dispatch_alert": dispatch,
            "details": risk_result["details"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# If run directly (not via uvicorn CLI), start the server
if __name__ == "__main__":
    import uvicorn
    print("Starting CodeRed ML Microservice on port 8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
