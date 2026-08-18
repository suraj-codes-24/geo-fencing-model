# CodeRed Geofencing & Risk Model - Context & Progress

## 1. Project Context
This project is a decoupled spatial ML module for the **Tourist Safety Portal (CodeRed)**.
The objective is to build an intelligent geofencing system that predicts risk levels across a city and dynamically draws danger polygons, rather than relying on arbitrary radius alerts.

## 2. Locked Architecture
*   **Spatial Unit:** H3 Grid System (Resolution 9, approx 0.1 sq km).
*   **Time Unit:** Hourly.
*   **Prediction Model:** 4 independent XGBoost Regressors predicting continuous scores (0-100).
*   **Categories:**
    1.  Crime & Security
    2.  Road & Accident
    3.  Environmental
    4.  Isolation & Emergency Access
*   **Classification:** SAFE (0-39), CAUTION (40-69), CRITICAL (70-100).
*   **Geofencing Logic:** Contiguous 'Critical' H3 cells are merged into a GeoJSON polygon with a 200m warning buffer.
*   **Location:** Kanpur City & PSIT campus corridor.

## 3. Current Progress
*   [x] **Problem Statement Analysis:** Discussed and aligned on the 3-layer architecture (ML Prediction -> Spatial Engine -> Geofence Engine).
*   [x] **Design Specification:** Created `dataset_design_spec.md` with advanced features (CCTV, Tourist Footfall, Terrain) and exact latent mathematical formulas for synthetic target generation.
*   [x] **Spatial Verification:** Generated `kanpur_grid_map.html` verifying that the bounding box correctly divides Kanpur into 10,453 H3 hexagons.

## 4. Next Immediate Steps
1.  **Generate Synthetic Data:** Write and execute `generate_data.py` step-by-step to produce the ~7.5 million row CSV dataset.
2.  **Train ML Models:** Train the 4 XGBoost regressors on the generated data.
3.  **Build Spatial Engine:** Implement the H3 cell merging and geofence polygon logic.
4.  **Local Visualization:** Build a local dashboard or API to visualize predictions on a map.
