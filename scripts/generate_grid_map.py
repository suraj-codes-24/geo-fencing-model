import h3
import folium
from shapely.geometry import Polygon

# 1. Define the exact bounding box for Kanpur + PSIT
MIN_LAT, MAX_LAT = 26.3100, 26.5100
MIN_LNG, MAX_LNG = 79.9100, 80.4000
RESOLUTION = 9 # Approx 0.1 sq km per hex

def get_hexes_in_bbox(min_lat, max_lat, min_lng, max_lng, res):
    """
    Finds all H3 hexagons that cover the given bounding box.
    """
    # Create a polygon for the bounding box
    bbox_geo = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lng, min_lat],
                [min_lng, max_lat],
                [max_lng, max_lat],
                [max_lng, min_lat],
                [min_lng, min_lat]
            ]
        ]
    }
    
    # geo_to_cells fills a geojson polygon with H3 hexes
    hexes = h3.geo_to_cells(bbox_geo, res)
    return list(hexes)

print("Generating H3 Grid for Kanpur...")
kanpur_hexes = get_hexes_in_bbox(MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG, RESOLUTION)
print(f"Generated {len(kanpur_hexes)} sectors (hexagons) for Kanpur City & PSIT.")

# 2. Create an interactive Map to visualize the sectors
print("Plotting sectors on an interactive map...")
map_center = [(MIN_LAT + MAX_LAT) / 2, (MIN_LNG + MAX_LNG) / 2]
m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

# Draw the bounding box for reference
folium.Rectangle(
    bounds=[[MIN_LAT, MIN_LNG], [MAX_LAT, MAX_LNG]],
    color='#ff0000',
    fill=False,
    weight=2,
    dash_array='5, 5'
).add_to(m)

# Draw all the hexagons
for hex_id in kanpur_hexes:
    # Get the boundaries of the hex
    hex_boundary = h3.cell_to_boundary(hex_id)
    
    # Plot polygon on the map
    folium.Polygon(
        locations=hex_boundary,
        color='#3388ff',
        weight=1,
        fill=True,
        fill_color='#3388ff',
        fill_opacity=0.1
    ).add_to(m)

# Add our Key Landmarks (Anchors) as markers
landmarks = {
    "PSIT Kanpur": (26.3367, 79.9290),
    "Kanpur Central Railway Station": (26.4540, 80.3496),
    "Z Square Mall": (26.4678, 80.3493),
    "Ganga Barrage": (26.4950, 80.3150),
    "Fazalganj Industrial": (26.4510, 80.3200)
}

for name, coords in landmarks.items():
    folium.Marker(
        location=coords,
        popup=name,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

# Save the map to an HTML file so the user can open it
output_file = "kanpur_grid_map.html"
m.save(output_file)
print(f"Success! Open '{output_file}' in your web browser to see the Kanpur sectors.")
