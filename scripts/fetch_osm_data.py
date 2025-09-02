import osmnx as ox
import geopandas as gpd
import yaml
from pathlib import Path

# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print("Loaded config:", config)
    print("File path:", Path(__file__).parent.parent / "config.yaml")

def fetch_data():
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    place_name = config["area"]["place_name"]

    # 1. Fetch roads and infer sidewalks from road tags
    print("1/6 Fetching roads and inferred sidewalks...")
    roads = ox.graph_from_place(place_name, network_type="drive")
    roads_gdf = ox.graph_to_gdfs(roads, nodes=False, edges=True)
    
    # Roads that likely have sidewalks
    roads_with_sidewalks = ox.features_from_place(
        place_name,
        tags={"highway": ["residential", "tertiary", "secondary"], "foot": "yes"}
    )

    # 2. Comprehensive sidewalk data
    print("2/6 Fetching detailed sidewalks...")
    sidewalks = ox.features_from_place(
        place_name,
        tags={
            "highway": ["footway", "path"],
            "footway": ["sidewalk", "crossing"],
            "foot": "designated"
        }
    )

    # 3. Crosswalks and pedestrian crossings
    print("3/6 Fetching crosswalks...")
    crosswalks = ox.features_from_place(
        place_name,
        tags={
            "highway": "crossing",
            "crossing": ["marked", "unmarked", "traffic_signals"],
            "crossing:markings": ["zebra", "lines"]
        }
    )

    # 4. Curb ramps (accessibility)
    print("4/6 Fetching curb ramps...")
    curb_ramps = ox.features_from_place(
        place_name,
        tags={"kerb": "lowered", "tactile_paving": "yes"}
    )

    # 5. Pedestrian zones
    print("5/6 Fetching pedestrian zones...")
    ped_zones = ox.features_from_place(
        place_name,
        tags={"highway": "pedestrian"}
    )

    # 6. Buildings
    print("6/6 Fetching buildings...")
    buildings = ox.features_from_place(
        place_name,
        tags={"building": True}
    )

    # Save all data
    print("Saving data...")
    roads_gdf.to_file(outputs_dir / "raw_roads.geojson", driver="GeoJSON")
    sidewalks.to_file(outputs_dir / "raw_sidewalks.geojson", driver="GeoJSON")
    crosswalks.to_file(outputs_dir / "raw_crosswalks.geojson", driver="GeoJSON")
    curb_ramps.to_file(outputs_dir / "raw_curb_ramps.geojson", driver="GeoJSON")
    ped_zones.to_file(outputs_dir / "raw_pedestrian_zones.geojson", driver="GeoJSON")
    buildings.to_file(outputs_dir / "raw_buildings.geojson", driver="GeoJSON")

    print("✅ All data saved to /outputs")

if __name__ == "__main__":
    fetch_data()
