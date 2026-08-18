import requests
import json
import time

API_URL = "http://127.0.0.1:8000/api/v1/geofence/check"

def test_ping(user_id, lat, lng, description):
    print(f"\n--- {description} ---")
    payload = {
        "user_id": user_id,
        "lat": lat,
        "lng": lng
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']} | Dispatch Alert: {data['dispatch_alert']}")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Testing CodeRed FastAPI Microservice...")
    
    # Ping 1: PSIT Campus (Should be CAUTION, so dispatch=False)
    test_ping("tourist_1", 26.3367, 79.9290, "Ping 1: PSIT Campus (CAUTION)")
    
    # Ping 2: Ganga Barrage (Depending on time, might be SAFE or CAUTION)
    test_ping("tourist_1", 26.4950, 80.3150, "Ping 2: Ganga Barrage")
    
    print("\nAPI Test Complete!")
