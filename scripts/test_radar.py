import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_risk, RESOLUTION
import h3

lat, lng = 26.4950, 80.3150
center_cell = h3.latlng_to_cell(lat, lng, RESOLUTION)
nearby_cells = h3.grid_disk(center_cell, 15)

highest = 0
for h in nearby_cells:
    c_lat, c_lng = h3.cell_to_latlng(h)
    res = predict_risk(c_lat, c_lng)
    if res["overall_score"] > highest:
        highest = res["overall_score"]
        
print(f"Highest score found in radar: {highest}")
