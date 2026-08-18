import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_risk
import numpy as np

# Kanpur Bounding Box
MIN_LAT, MAX_LAT = 26.31, 26.51
MIN_LNG, MAX_LNG = 79.91, 80.40

print("Scanning for CRITICAL hotspots right now...")

# Create a coarse grid to find a fast hotspot
lats = np.linspace(MIN_LAT, MAX_LAT, 20)
lngs = np.linspace(MIN_LNG, MAX_LNG, 20)

found = False
for lat in lats:
    for lng in lngs:
        res = predict_risk(lat, lng)
        if res["overall_score"] >= 70:
            print(f"CRITICAL ZONE FOUND! -> Lat: {lat:.5f}, Lng: {lng:.5f}")
            print(f"Details: {res['details']}")
            found = True
            break
    if found:
        break

if not found:
    print("No purely critical zones found. Finding highest risk zone...")
    highest = 0
    best_coords = (0,0)
    for lat in lats:
        for lng in lngs:
            res = predict_risk(lat, lng)
            if res["overall_score"] > highest:
                highest = res["overall_score"]
                best_coords = (lat, lng)
    print(f"Highest Risk Zone -> Lat: {best_coords[0]:.5f}, Lng: {best_coords[1]:.5f} with Score: {highest:.2f}")
