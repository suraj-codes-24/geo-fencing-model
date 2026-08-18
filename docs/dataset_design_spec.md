# Comprehensive Data Generation Design Specification

This document details the precise mathematical logic, spatial distribution rules, and schema definitions that will be used to generate the synthetic dataset for the CodeRed Geofencing & Risk Model in Kanpur.

## 1. Spatial Domain & Grid System

We are using Uber's H3 spatial indexing system at **Resolution 9**.
*   **Average Hexagon Area:** ~0.105 sq km.
*   **Bounding Box (Kanpur City to PSIT):**
    *   Latitude: `26.3000` to `26.5000`
    *   Longitude: `79.9000` to `80.4000`
*   **Methodology:** We will generate a grid covering this bounding box. Any H3 cells whose centroids fall outside the box will be excluded.

## 2. Simulated Hotspots (Anchors)

To create realistic spatial gradients (e.g., distance to police, density of traffic), we define specific geographical anchors within Kanpur.

| Anchor Name | Lat/Lng | Characteristics |
| :--- | :--- | :--- |
| **Kanpur Central Railway Station** | `26.4540, 80.3496` | High crime baseline, very high traffic, 24/7 footfall, very close to police, high CCTV. |
| **Z Square Mall Area** | `26.4678, 80.3493` | High traffic density, low night crime (good lighting, high CCTV), high daytime tourist/commercial density. |
| **Ganga Barrage** | `26.4950, 80.3150` | High environment risk (flood/water), lower lighting, moderate isolation, river_bank terrain. |
| **PSIT (Institute)** | `26.3367, 79.9290` | Campus environment, surrounded by highway. High traffic speed, low density. |
| **Fazalganj Industrial Estate** | `26.4510, 80.3200` | High AQI/PM2.5, low weekend traffic, higher isolation at night, industrial terrain. |

Distance to these anchors will be calculated for every H3 cell to determine its feature baselines.

## 3. Temporal Domain

*   **Granularity:** Hourly (1-hour time steps).
*   **Duration:** 30 continuous days.
*   **Seasonality Simulated:** We will simulate a standard month (e.g., occasional rain spikes, standard weekday/weekend traffic patterns).

## 4. Feature Schema & Data Types

Below is the exact schema for the output CSV, now enriched with advanced Tourist & Infrastructure dimensions.

| Column Name | Data Type | Range/Possible Values | Description |
| :--- | :--- | :--- | :--- |
| `h3_cell` | String | e.g. `8928308280fffff` | The H3 index of the area. |
| `timestamp` | Datetime | `YYYY-MM-DD HH:MM:SS` | The hourly timestamp. |
| `hour_of_day` | Integer | `0` to `23` | Extracted hour. |
| `day_of_week` | Integer | `0` (Mon) to `6` (Sun) | Extracted day. |
| `is_weekend` | Integer | `0` or `1` | Binary flag. |
| `is_festival_day` | Integer | `0` or `1` | Simulates days with massive public gatherings. |
| `terrain_type` | Categorical | `urban`, `highway`, `river`, `industrial` | The geographic nature of the cell. |
| `cctv_coverage_score` | Float | `0.0` to `100.0` | High in malls/stations, near 0 on highways/outskirts. |
| `tourist_footfall_density`| Float | `0.0` to `100.0` | High near malls, barrage, stations. Low in industrial. |
| `commercial_density` | Float | `0.0` to `100.0` | Concentration of shops/restaurants. |
| `crime_count_historical`| Integer | `0` to `50` | Baseline crime for this cell based on proximity to Central/Malls. |
| `reported_harassment_cases`| Integer| `0` to `15` | Tourist-specific scam/harassment incidents. |
| `recent_crime_count` | Integer | `0` to `10` | Random spikes simulating recent incidents. |
| `night_crime_ratio` | Float | `0.0` to `1.0` | Higher in industrial/outskirt cells. |
| `avg_crime_severity` | Float | `1.0` to `10.0` | 1=Petty theft, 10=Violent crime. |
| `traffic_density` | Float | `0.0` to `100.0` | Peaks at 9AM and 6PM. High near malls/station. |
| `road_hazard_score` | Float | `0.0` to `100.0` | Static per cell. Simulates bad junctions/potholes. |
| `road_lighting_score`| Float | `0.0` to `100.0` | High in city, drops near highway/PSIT at night. |
| `historical_accident_count`| Integer| `0` to `20` | Correlated with high traffic + high hazard. |
| `aqi` | Float | `50.0` to `400.0`| Peaks in morning/evening, highest near Fazalganj. |
| `rainfall_mm` | Float | `0.0` to `50.0` | Random weather events affecting visibility. |
| `flood_risk_score` | Float | `0.0` to `100.0` | Peaks near Ganga Barrage. |
| `police_distance_km`| Float | `0.1` to `20.0` | Euclidean/Haversine distance to nearest simulated station. |
| `hospital_distance_km`| Float | `0.1` to `20.0` | Euclidean/Haversine distance to nearest hospital. |
| `ambulance_reach_time_mins`| Float | `2.0` to `60.0` | function of (`hospital_distance` * `traffic_density`). |
| `network_score` | Float | `0.0` to `100.0` | Signal strength. Lower in outskirts/highways. |

## 5. Explicit Latent Target Generation Formulas

This is the mathematical core of the simulation. The XGBoost models will attempt to learn these relationships. We add Gaussian noise $\mathcal{N}(0, \sigma)$ to prevent the model from getting a perfect score.

### A. Crime Risk (0-100)
```text
Base = (historical_crime * 1.2) + (recent_crime * 2.5) + avg_crime_severity + (reported_harassment_cases * 1.5)
Time_Penalty = IF(hour_of_day >= 22 OR hour_of_day <= 4, night_crime_ratio * 15, 0)
Lighting_Penalty = (100 - road_lighting_score) * 0.15
Crowd_Factor = IF(is_festival_day == 1, tourist_footfall_density * 0.2, 0)  # High crowds = pickpocketing
CCTV_Deterrent = cctv_coverage_score * 0.2  # Reduces risk

Raw_Crime = Base + Time_Penalty + Lighting_Penalty + Crowd_Factor - CCTV_Deterrent + Noise(mean=0, std=5)
Crime_Risk = MIN(MAX(Raw_Crime, 0), 100)
```

### B. Accident Risk (0-100)
```text
Base = (traffic_density * 0.35) + (road_hazard_score * 0.3)
Terrain_Risk = IF(terrain_type == 'highway', 15, 0) # High speed risk
History_Weight = historical_accident_count * 1.2
Weather_Penalty = (rainfall_mm * 1.2) + IF(road_lighting_score < 40 AND hour >= 18, 15, 0)
Festival_Traffic = IF(is_festival_day == 1, 10, 0)

Raw_Accident = Base + Terrain_Risk + History_Weight + Weather_Penalty + Festival_Traffic + Noise(mean=0, std=7)
Accident_Risk = MIN(MAX(Raw_Accident, 0), 100)
```

### C. Environment Risk (0-100)
```text
AQI_Factor = (aqi - 50) * 0.2  
Flood_Factor = flood_risk_score * (1 + (rainfall_mm / 10))
Terrain_Risk = IF(terrain_type == 'river', 10, 0)

Raw_Env = AQI_Factor + Flood_Factor + Terrain_Risk + Noise(mean=0, std=4)
Environment_Risk = MIN(MAX(Raw_Env, 0), 100)
```

### D. Isolation Risk (0-100)
```text
Emergency_Distance = (police_distance_km * 1.5)
Ambulance_Factor = ambulance_reach_time_mins * 0.5
Network_Penalty = (100 - network_score) * 0.5
Time_Penalty = IF(hour_of_day >= 22 OR hour_of_day <= 5, 10, 0)

Raw_Isolation = Emergency_Distance + Ambulance_Factor + Network_Penalty + Time_Penalty + Noise(mean=0, std=3)
Isolation_Risk = MIN(MAX(Raw_Isolation, 0), 100)
```

### E. The Final Composite Ground Truth (For Reference)
```text
Overall_Risk = (0.35 * Crime_Risk) + (0.25 * Accident_Risk) + (0.20 * Environment_Risk) + (0.20 * Isolation_Risk)
```

## 6. Execution Flow for Data Generation

1.  **Grid Initialization:** Generate all H3 Resolution 9 hexes within the Kanpur BBox.
2.  **Static Feature Mapping:** Assign static values (`police_distance_km`, `terrain_type`, `cctv_coverage_score`, etc.) based on hex centroid distance to the Anchors.
3.  **Temporal Expansion:** Cross-join the grid with a date/time sequence (720 hours for 30 days).
4.  **Dynamic Feature Calculation:** Apply time-based curves (e.g., `traffic_density` peaks at 9AM and 6PM, festival days trigger crowds).
5.  **Target Calculation:** Apply the Latent Formulas (Section 5) to generate the target risks.
6.  **CSV Export:** Save to `kanpur_synthetic_dataset.csv`.
