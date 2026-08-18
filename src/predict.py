import os
import time
import math
import joblib
import pandas as pd
import numpy as np
import h3
import folium
from datetime import datetime, timedelta
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
ANCHORS = {
    "Kanpur_Central": {"lat": 26.4540, "lng": 80.3496, "type": "urban"},
    "Z_Square_Mall": {"lat": 26.4678, "lng": 80.3493, "type": "urban"},
    "Ganga_Barrage": {"lat": 26.4950, "lng": 80.3150, "type": "river"},
    "PSIT": {"lat": 26.3367, "lng": 79.9290, "type": "highway"},
    "Fazalganj": {"lat": 26.4510, "lng": 80.3200, "type": "industrial"}
}

MIN_LAT, MAX_LAT = 26.3100, 26.5100
MIN_LNG, MAX_LNG = 79.9100, 80.4000
RESOLUTION = 9 

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("Loading ML Models and Feature Schema...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model_crime = joblib.load(os.path.join(BASE_DIR, "models", "xgb_crime.pkl"))
    model_accident = joblib.load(os.path.join(BASE_DIR, "models", "xgb_accident.pkl"))
    model_env = joblib.load(os.path.join(BASE_DIR, "models", "xgb_environment.pkl"))
    model_iso = joblib.load(os.path.join(BASE_DIR, "models", "xgb_isolation.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns.pkl"))
except Exception as e:
    print(f"Error loading models: {e}. Please ensure 'train_models.py' ran successfully.")
    exit(1)

# ==========================================
# 3. FEATURE ENGINEERING ENGINE
# ==========================================
def build_features_for_location(lat: float, lng: float, current_time: datetime = None) -> pd.DataFrame:
    if current_time is None:
        current_time = datetime.now()
        
    dists = {name: haversine(lat, lng, loc["lat"], loc["lng"]) for name, loc in ANCHORS.items()}
    closest_anchor = min(dists, key=dists.get)
    terrain_type = ANCHORS[closest_anchor]["type"]
    dist_to_center = min(dists["Kanpur_Central"], dists["Z_Square_Mall"])
    
    police_dist = max(0.1, dist_to_center * 1.2)
    hospital_dist = max(0.5, dist_to_center * 1.5)
    cctv = max(0, 100 - (dist_to_center * 15))
    base_crime = max(0, 40 - (dist_to_center * 3))
    lighting = max(20, 100 - (dist_to_center * 5))
    hazard = min(100, 20 + (dists["Fazalganj"] * 2))
    network = max(10, 100 - (dists["Kanpur_Central"] * 2))
    flood_risk = max(0, 100 - (dists["Ganga_Barrage"] * 10))
    
    hour = current_time.hour
    is_weekend = 1 if current_time.weekday() >= 5 else 0
    is_festival = 0
    
    base_traffic = 20 if is_weekend else 40
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        base_traffic += 50
    decay = max(0, 1 - (dist_to_center / 15))
    traffic = min(100, base_traffic * decay)
    
    footfall = max(0, 80 - (dist_to_center * 10)) if 10 <= hour <= 20 else 5
    
    feature_dict = {
        'hour_of_day': hour,
        'day_of_week': current_time.weekday(),
        'is_weekend': is_weekend,
        'is_festival_day': is_festival,
        'cctv_coverage_score': cctv,
        'tourist_footfall_density': footfall,
        'crime_count_historical': base_crime,
        'reported_harassment_cases': footfall * 0.05,
        'recent_crime_count': 0, 
        'night_crime_ratio': 0.8 if (hour >= 22 or hour <= 4) else 0.1,
        'avg_crime_severity': 5.0, 
        'traffic_density': traffic,
        'road_hazard_score': hazard,
        'road_lighting_score': lighting,
        'aqi': 100 + (traffic * 0.5),
        'rainfall_mm': 0.0, 
        'flood_risk_score': flood_risk,
        'police_distance_km': police_dist,
        'hospital_distance_km': hospital_dist,
        'ambulance_reach_time_mins': hospital_dist * (1 + (traffic / 50)) * 2,
        'network_score': network,
        'terrain_type_highway': 1 if terrain_type == 'highway' else 0,
        'terrain_type_industrial': 1 if terrain_type == 'industrial' else 0,
        'terrain_type_river': 1 if terrain_type == 'river' else 0,
        'terrain_type_urban': 1 if terrain_type == 'urban' else 0,
    }
    
    df = pd.DataFrame([feature_dict])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

def predict_risk(lat: float, lng: float, current_time: datetime = None) -> dict:
    # Build features
    features_df = build_features_for_location(lat, lng, current_time)
    
    # --- DEMO OVERRIDE ---
    # To ensure the user can actually see Geofences trigger in the UI right now,
    # we simulate a Severe Rainstorm, Festival crowd, and Evening Rush Hour.
    features_df["rainfall_mm"] = 45.0
    features_df["is_festival_day"] = 1
    features_df["hour_of_day"] = 18
    features_df["traffic_density"] = 100
    features_df["tourist_footfall_density"] = 100
    
    # Predict sub-risks
    crime = float(model_crime.predict(features_df)[0])
    acc = float(model_accident.predict(features_df)[0])
    env = float(model_env.predict(features_df)[0])
    iso = float(model_iso.predict(features_df)[0])
    overall = (0.35 * crime) + (0.25 * acc) + (0.20 * env) + (0.20 * iso)
    
    if overall >= 70:
        status = "CRITICAL"
    elif overall >= 40:
        status = "CAUTION"
    else:
        status = "SAFE"
        
    return {
        "overall_score": round(overall, 2),
        "status": status,
        "details": {
            "Crime_Risk": round(crime, 2),
            "Accident_Risk": round(acc, 2),
            "Environment_Risk": round(env, 2),
            "Isolation_Risk": round(iso, 2)
        }
    }

def get_nearby_danger_zones(lat: float, lng: float, radius_k: int = 15) -> dict:
    """
    Scans an approx 3km radius (k=15 at resolution 9) around the given point.
    Returns a GeoJSON dict of merged critical polygons.
    """
    center_cell = h3.latlng_to_cell(lat, lng, RESOLUTION)
    nearby_cells = h3.grid_disk(center_cell, radius_k)
    
    critical_polygons = []
    
    current_time = datetime.now()
    for h in nearby_cells:
        c_lat, c_lng = h3.cell_to_latlng(h)
            
        res = predict_risk(c_lat, c_lng, current_time)
        if res["overall_score"] >= 70:
            boundary = h3.cell_to_boundary(h)
            shapely_coords = [(blng, blat) for blat, blng in boundary]
            critical_polygons.append(Polygon(shapely_coords))
            
    if not critical_polygons:
        return None
        
    merged_danger_zones = unary_union(critical_polygons)
    WARNING_BUFFER_DEGREES = 0.0018 # ~200m
    warning_zones = merged_danger_zones.buffer(WARNING_BUFFER_DEGREES, join_style=2)
    
    # Convert Shapely MultiPolygon/Polygon to GeoJSON
    return mapping(warning_zones)

# ==========================================
# 4. HYSTERESIS MANAGER (LAYER 3)
# ==========================================
class GeofenceHysteresisManager:
    """Prevents notification spam using time-based hysteresis."""
    def __init__(self, cooldown_minutes=5):
        self.active_alerts = {} # user_id -> timestamp
        self.cooldown = timedelta(minutes=cooldown_minutes)
        
    def should_trigger_alert(self, user_id: str, status: str, current_time: datetime) -> bool:
        if status != "CRITICAL":
            return False # Only alert on critical
            
        last_alert = self.active_alerts.get(user_id)
        if last_alert is None or (current_time - last_alert) > self.cooldown:
            self.active_alerts[user_id] = current_time
            return True
            
        return False # Suppressed due to hysteresis

# ==========================================
# 5. GEOFENCE MAP GENERATOR (LAYER 2)
# ==========================================
def scan_city_and_map():
    print(f"\n--- Scanning Kanpur City for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    bbox_geo = {
        "type": "Polygon",
        "coordinates": [[[MIN_LNG, MIN_LAT], [MIN_LNG, MAX_LAT], [MAX_LNG, MAX_LAT], [MAX_LNG, MIN_LAT], [MIN_LNG, MIN_LAT]]]
    }
    hexes = list(h3.geo_to_cells(bbox_geo, RESOLUTION))
    
    critical_polygons = []
    
    for h in hexes:
        lat, lng = h3.cell_to_latlng(h)
        # Demo Spikes: Simulate a rainstorm and festival just to guarantee we see polygons on the map
        features_df = build_features_for_location(lat, lng)
        features_df["rainfall_mm"] = 30.0
        features_df["is_festival_day"] = 1
        
        crime = float(model_crime.predict(features_df)[0])
        acc = float(model_accident.predict(features_df)[0])
        env = float(model_env.predict(features_df)[0])
        iso = float(model_iso.predict(features_df)[0])
        overall = (0.35 * crime) + (0.25 * acc) + (0.20 * env) + (0.20 * iso)
        
        # Using 50 threshold instead of 70 just to guarantee polygons show up for the prototype demo
        if overall >= 50:
            # h3.cell_to_boundary returns ((lat, lng), ...) 
            # Shapely requires ((lng, lat), ...)
            boundary = h3.cell_to_boundary(h)
            shapely_coords = [(lng, lat) for lat, lng in boundary]
            critical_polygons.append(Polygon(shapely_coords))
            
    print(f"Found {len(critical_polygons)} at-risk hexagons. Merging into contiguous polygons...")
    
    # Initialize Map
    map_center = [(MIN_LAT + MAX_LAT) / 2, (MIN_LNG + MAX_LNG) / 2]
    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB dark_matter")
    
    if len(critical_polygons) > 0:
        # Layer 2: Merge adjacent polygons using Shapely
        merged_danger_zones = unary_union(critical_polygons)
        
        # Layer 2: Create a 200m buffer
        # Roughly 1 degree of lat/lng is 111km. So 200m is ~0.0018 degrees.
        WARNING_BUFFER_DEGREES = 0.0018
        warning_zones = merged_danger_zones.buffer(WARNING_BUFFER_DEGREES, join_style=2)
        
        # Helper to plot Shapely objects in Folium
        def plot_shapely(geom, color, fill_opacity, popup):
            if geom.geom_type == 'Polygon':
                geoms = [geom]
            elif geom.geom_type == 'MultiPolygon':
                geoms = list(geom.geoms)
            else:
                return
                
            for g in geoms:
                if not g.exterior or len(g.exterior.coords) == 0:
                    continue
                # Shapely is (lng, lat), Folium needs (lat, lng)
                folium_coords = [(lat, lng) for lng, lat in g.exterior.coords]
                folium.Polygon(
                    locations=folium_coords,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=popup
                ).add_to(m)

        # 1. Plot the Yellow Warning Buffer first (so it goes underneath)
        plot_shapely(warning_zones, 'yellow', 0.2, "WARNING ZONE (200m Buffer)")
        
        # 2. Plot the Red Critical Core
        plot_shapely(merged_danger_zones, 'red', 0.5, "CRITICAL ZONE")
            
    output_file = "live_geofence_map.html"
    m.save(output_file)
    print(f"Geofence Map with Shapely merging & buffers saved to '{output_file}'!")

# ==========================================
# DEMO EXECUTION
# ==========================================
if __name__ == "__main__":
    # Test 1: Hysteresis Logic
    print("\n--- HYSTERESIS (LAYER 3) TEST ---")
    manager = GeofenceHysteresisManager(cooldown_minutes=5)
    now = datetime.now()
    
    print("Ping 1 (CRITICAL):", "ALERT SENT!" if manager.should_trigger_alert("Tourist_A", "CRITICAL", now) else "Suppressed")
    print("Ping 2 (CRITICAL - 1 min later):", "ALERT SENT!" if manager.should_trigger_alert("Tourist_A", "CRITICAL", now + timedelta(minutes=1)) else "Suppressed (Hysteresis Active)")
    print("Ping 3 (CRITICAL - 6 min later):", "ALERT SENT!" if manager.should_trigger_alert("Tourist_A", "CRITICAL", now + timedelta(minutes=6)) else "Suppressed")

    # Test 2: Full Map Gen with Shapely
    scan_city_and_map()
