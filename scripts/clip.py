import geopandas as gpd
import yaml
from pathlib import Path
from shapely.geometry import box
import logging

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Load Config --------------------
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

outputs_dir = Path(__file__).parent.parent / "outputs"
clipped_dir = outputs_dir

# Bounding box (WGS84)
bbox_cfg = config["area"]["bbox"]
clip_box = box(bbox_cfg["west"], bbox_cfg["south"], bbox_cfg["east"], bbox_cfg["north"])

# Raw files to clip
raw_files = [
    "raw_roads.geojson",
    "raw_sidewalks.geojson",
    "raw_crosswalks.geojson",
    "raw_curb_ramps.geojson",
    "raw_buildings.geojson",
    "raw_pedestrian_zones.geojson"
]

# -------------------- Clip Loop --------------------
for fname in raw_files:
    file_path = outputs_dir / fname
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        continue

    logger.info(f"Clipping {fname}...")
    gdf = gpd.read_file(file_path)

    # Ensure CRS matches bounding box CRS
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Clip by intersection
    gdf_clipped = gdf[gdf.geometry.intersects(clip_box)].copy()

    # Save clipped file separately
    clipped_path = clipped_dir / fname
    gdf_clipped.to_file(clipped_path, driver="GeoJSON")
    logger.info(f"Saved clipped file: {clipped_path.name}")

logger.info("✅ All raw files clipped to bounding box")
