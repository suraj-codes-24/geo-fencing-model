import pandas as pd
import numpy as np
import h3
import math
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# ==========================================
# PHASE 1: CONFIGURATION & SPATIAL ANCHORS
# ==========================================
MIN_LAT, MAX_LAT = 26.3100, 26.5100
MIN_LNG, MAX_LNG = 79.9100, 80.4000
RESOLUTION = 9 
SIMULATION_DAYS = 14 # Generating 14 days of data.

ANCHORS = {
    "Kanpur_Central": {"lat": 26.4540, "lng": 80.3496, "type": "urban"},
    "Z_Square_Mall": {"lat": 26.4678, "lng": 80.3493, "type": "urban"},
    "Ganga_Barrage": {"lat": 26.4950, "lng": 80.3150, "type": "river"},
    "PSIT": {"lat": 26.3367, "lng": 79.9290, "type": "highway"},
    "Fazalganj": {"lat": 26.4510, "lng": 80.3200, "type": "industrial"}
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lng points."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_kanpur_grid() -> List[str]:
    """Generates all H3 hexes within the bounding box."""
    bbox_geo = {
        "type": "Polygon",
        "coordinates": [[
            [MIN_LNG, MIN_LAT], [MIN_LNG, MAX_LAT], 
            [MAX_LNG, MAX_LAT], [MAX_LNG, MIN_LAT], [MIN_LNG, MIN_LAT]
        ]]
    }
    return list(h3.geo_to_cells(bbox_geo, RESOLUTION))

def build_spatial_base() -> pd.DataFrame:
    print("Phase 1: Generating Spatial Grid and Static Features...")
    hexes = get_kanpur_grid()
    data = []
    
    for h in hexes:
        lat, lng = h3.cell_to_latlng(h)
        
        # Calculate distances to all anchors
        dists = {name: haversine(lat, lng, loc["lat"], loc["lng"]) for name, loc in ANCHORS.items()}
        
        # Determine Terrain Type based on closest anchor
        closest_anchor = min(dists, key=dists.get)
        terrain_type = ANCHORS[closest_anchor]["type"]
        
        # Police/Hospital Distances (simulated based on Central & Malls)
        dist_to_center = min(dists["Kanpur_Central"], dists["Z_Square_Mall"])
        police_dist = max(0.1, dist_to_center * 1.2 + np.random.uniform(0, 1))
        hospital_dist = max(0.5, dist_to_center * 1.5 + np.random.uniform(0, 2))
        
        # CCTV Coverage (High near malls/station, drops rapidly further away)
        cctv = max(0, 100 - (dist_to_center * 15))
        
        # Base Crime (Higher near central station, lower near PSIT)
        base_crime = max(0, 40 - (dist_to_center * 3))
        
        # Hazard & Light (Better in city, worse on highway/outskirts)
        lighting = max(20, 100 - (dist_to_center * 5))
        hazard = min(100, 20 + (dists["Fazalganj"] * 2) + np.random.uniform(0, 10))
        
        # Network & Flood
        network = max(10, 100 - (dists["Kanpur_Central"] * 2))
        flood_risk = max(0, 100 - (dists["Ganga_Barrage"] * 10))
        
        data.append({
            "h3_cell": h,
            "terrain_type": terrain_type,
            "dist_to_center": dist_to_center,
            "cctv_coverage_score": round(cctv, 2),
            "crime_count_historical": int(base_crime),
            "road_lighting_score": round(lighting, 2),
            "road_hazard_score": round(hazard, 2),
            "flood_risk_score": round(flood_risk, 2),
            "police_distance_km": round(police_dist, 2),
            "hospital_distance_km": round(hospital_dist, 2),
            "network_score": round(network, 2)
        })
        
    return pd.DataFrame(data)

# ==========================================
# PHASE 2: TEMPORAL EXPANSION
# ==========================================
def expand_temporally(spatial_df: pd.DataFrame, days: int) -> pd.DataFrame:
    print(f"Phase 2: Expanding Data for {days} Days (Hourly)...")
    start_date = datetime(2026, 8, 1)
    hours = days * 24
    
    time_index = [start_date + timedelta(hours=i) for i in range(hours)]
    time_df = pd.DataFrame({"timestamp": time_index})
    
    # Extract time features
    time_df["hour_of_day"] = time_df["timestamp"].dt.hour
    time_df["day_of_week"] = time_df["timestamp"].dt.dayofweek
    time_df["is_weekend"] = time_df["day_of_week"].isin([5, 6]).astype(int)
    
    # Simulate a festival day on the 5th day of the simulation
    fest_date = (start_date + timedelta(days=4)).date()
    time_df["is_festival_day"] = (time_df["timestamp"].dt.date == fest_date).astype(int)
    
    # Cross join (Cartesian product)
    spatial_df["key"] = 1
    time_df["key"] = 1
    full_df = pd.merge(time_df, spatial_df, on="key").drop("key", axis=1)
    
    return full_df

# ==========================================
# PHASE 3: DYNAMIC FEATURES
# ==========================================
def calculate_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Phase 3: Calculating Dynamic Time-Series Features...")
    
    # Traffic peaks at 9AM and 6PM
    def traffic_curve(hour, is_weekend, dist):
        base = 20 if is_weekend else 40
        if 8 <= hour <= 10 or 17 <= hour <= 19:
            base += 50
        # Traffic is denser closer to the city center
        decay = max(0, 1 - (dist / 15))
        return min(100, base * decay + np.random.uniform(0, 10))
    
    df["traffic_density"] = np.vectorize(traffic_curve)(df["hour_of_day"], df["is_weekend"], df["dist_to_center"])
    
    # Tourist Footfall (Active 10AM - 8PM, higher near malls)
    df["tourist_footfall_density"] = np.where(
        (df["hour_of_day"] >= 10) & (df["hour_of_day"] <= 20),
        np.maximum(0, 80 - (df["dist_to_center"] * 10)),
        5 # Night time baseline
    )
    
    # Weather & Environmental Simulation
    df["aqi"] = 100 + (df["traffic_density"] * 0.5) + np.random.normal(0, 15, len(df))
    # Random rain showers
    df["rainfall_mm"] = np.where(np.random.rand(len(df)) > 0.95, np.random.uniform(10, 40, len(df)), 0)
    
    # Dynamic Crime Modifiers
    df["night_crime_ratio"] = np.where((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 4), 0.8, 0.1)
    df["recent_crime_count"] = np.random.poisson(lam=0.5, size=len(df))
    df["avg_crime_severity"] = np.random.uniform(1, 10, len(df))
    df["reported_harassment_cases"] = (df["tourist_footfall_density"] * 0.05 * np.random.rand(len(df))).astype(int)
    
    # Emergency Response Speed
    df["ambulance_reach_time_mins"] = df["hospital_distance_km"] * (1 + (df["traffic_density"] / 50)) * 2
    
    return df

# ==========================================
# PHASE 4: TARGET FORMULAS
# ==========================================
def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    print("Phase 4: Calculating Latent Ground Truth Risks...")
    
    def bound(series): return np.clip(series, 0, 100)
    
    # Crime Risk
    crime_base = (df["crime_count_historical"] * 1.2) + (df["recent_crime_count"] * 2.5) + df["avg_crime_severity"] + (df["reported_harassment_cases"] * 1.5)
    crime_time = np.where((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 4), df["night_crime_ratio"] * 15, 0)
    crime_light = (100 - df["road_lighting_score"]) * 0.15
    crime_crowd = np.where(df["is_festival_day"] == 1, df["tourist_footfall_density"] * 0.2, 0)
    crime_cctv = df["cctv_coverage_score"] * 0.2
    df["Crime_Risk"] = bound(crime_base + crime_time + crime_light + crime_crowd - crime_cctv + np.random.normal(0, 5, len(df)))
    
    # Accident Risk
    acc_base = (df["traffic_density"] * 0.35) + (df["road_hazard_score"] * 0.3)
    acc_terrain = np.where(df["terrain_type"] == 'highway', 15, 0)
    acc_weather = (df["rainfall_mm"] * 1.2) + np.where((df["road_lighting_score"] < 40) & (df["hour_of_day"] >= 18), 15, 0)
    acc_fest = np.where(df["is_festival_day"] == 1, 10, 0)
    df["Accident_Risk"] = bound(acc_base + acc_terrain + acc_weather + acc_fest + np.random.normal(0, 7, len(df)))
    
    # Environment Risk
    env_aqi = (df["aqi"] - 50) * 0.2
    env_flood = df["flood_risk_score"] * (1 + (df["rainfall_mm"] / 10))
    env_terrain = np.where(df["terrain_type"] == 'river', 10, 0)
    df["Environment_Risk"] = bound(env_aqi + env_flood + env_terrain + np.random.normal(0, 4, len(df)))
    
    # Isolation Risk
    iso_dist = df["police_distance_km"] * 1.5
    iso_amb = df["ambulance_reach_time_mins"] * 0.5
    iso_net = (100 - df["network_score"]) * 0.5
    iso_time = np.where((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5), 10, 0)
    df["Isolation_Risk"] = bound(iso_dist + iso_amb + iso_net + iso_time + np.random.normal(0, 3, len(df)))
    
    # Composite (Optional Reference)
    df["Overall_Risk"] = (0.35 * df["Crime_Risk"]) + (0.25 * df["Accident_Risk"]) + (0.20 * df["Environment_Risk"]) + (0.20 * df["Isolation_Risk"])
    
    return df

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- STARTING SYNTHETIC DATA GENERATION ---")
    
    # Run the pipeline
    df_spatial = build_spatial_base()
    df_temporal = expand_temporally(df_spatial, days=SIMULATION_DAYS)
    df_dynamic = calculate_dynamic_features(df_temporal)
    df_final = calculate_targets(df_dynamic)
    
    # Downcast datatypes to save memory
    float_cols = df_final.select_dtypes(include=['float64']).columns
    int_cols = df_final.select_dtypes(include=['int64', 'int32']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')
    df_final[int_cols] = df_final[int_cols].astype('int16')
    
    output_file = "kanpur_synthetic_dataset.csv"
    print(f"Phase 5: Saving {len(df_final)} rows to {output_file}...")
    df_final.to_csv(output_file, index=False)
    
    print(f"Success! Data generation complete. Output size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
