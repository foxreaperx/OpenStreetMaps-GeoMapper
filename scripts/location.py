import requests
from geopy.geocoders import Nominatim

def get_current_location():
    """
    Get current lat/lon based on IP address (approximate).
    """
    try:
        res = requests.get("https://ipinfo.io/json")
        data = res.json()
        lat, lon = map(float, data["loc"].split(","))
        return lat, lon
    except Exception as e:
        raise RuntimeError(f"Could not get current location: {e}")

def build_bbox(lat, lon, block_size=0.002):
    """
    Build a bounding box around the given lat/lon.
    block_size ~0.002 degrees ≈ ~5 city blocks (depends on city).
    """
    north = lat + block_size
    south = lat - block_size
    east = lon + block_size
    west = lon - block_size
    return north, south, east, west

def get_area_info():
    lat, lon = get_current_location()
    
    # reverse geocode to get place name
    geolocator = Nominatim(user_agent="location_checker")
    location = geolocator.reverse((lat, lon), language="en")
    
    north, south, east, west = build_bbox(lat, lon)

    area_info = {
        "area": {
            "place_name": location.address if location else "Unknown location",
            "bbox": {
                "north": round(north, 6),
                "south": round(south, 6),
                "east": round(east, 6),
                "west": round(west, 6),
            },
        }
    }
    return area_info


if __name__ == "__main__":
    info = get_area_info()
    import yaml
    print(yaml.dump(info, sort_keys=False))
