# OpenStreetMaps-GeoMapper
# GeoJSON Generator for Pedestrian Infrastructure

This project generates **GeoJSON layers** of pedestrian and street infrastructure from **OpenStreetMap (OSM)**.  
It automates the workflow: **fetch raw OSM data → process geometries → output clean GeoJSON files** ready for accessibility analysis, urban design, or GIS visualization.

---

## ✨ Features

- **Fetch OSM data** for a chosen area (roads, sidewalks, crosswalks, curb ramps, pedestrian zones, buildings).  
- **Clean & fix geometries**: snap, merge, buffer, and segment streets.  
- **Generate polygons** for streets, sidewalks, and crosswalks with width inference and quality gates.  
- **Subtract overlaps** (e.g., crosswalks from roads).  
- **Sidewalk connectors**: automatically link crosswalks to sidewalks.  
- **Visualize** outputs via built-in scripts.  
- **Configurable** via `config.yaml` (CRS, widths, crosswalk rules).  
- **Outputs directly in GeoJSON** for easy use in GIS tools.  

---

## 🚀 Quickstart (User Guide)

### 1. Install dependencies
Requires **Python 3.9+**. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure project
Edit **`config.yaml`**, for example:

```yaml
area:
  place_name: "San Francisco, California, USA"
local_crs: "EPSG:32610"
buffers:
  road_width: 10.0
  sidewalk_width: 2.0
  crosswalk_width: 3.0
```

### 3. Fetch raw OSM data
```bash
python scripts/fetch_osm_data.py
```
Saves raw GeoJSONs into `outputs/` (roads, sidewalks, crosswalks, buildings, etc.).

### 4. Process data
```bash
python scripts/process_data.py
```
Generates cleaned **processed GeoJSON layers**: streets, sidewalks, crosswalks (with connectors), accessibility network.

### 5. Visualize
```bash
python scripts/visualize_data.py
```
Optional: display outputs for inspection may not be fully functional but can also just view online using geojson tool.

---

## 📂 Project Structure

```
Project/
└── project_updated/
    ├── config.yaml                # Configuration for place, CRS, widths, rules
    ├── requirements.txt           # Python dependencies
    ├── logs/                      # Run logs
    ├── outputs/                   # All GeoJSON outputs
    │   ├── raw_roads.geojson
    │   ├── raw_sidewalks.geojson
    │   ├── raw_crosswalks.geojson
    │   ├── raw_buildings.geojson
    │   ├── processed_accessibility.geojson
    │   ├── processed_pedestrian_network.geojson
    │   └── processed_streets_buildings.geojson
    ├── scripts/                   # Source code
    │   ├── fetch_osm_data.py      # Fetch raw OSM data
    │   ├── process_data.py        # Core processing pipeline
    │   ├── clip.py                # Clip datasets to bounding boxes
    │   ├── location.py            # Handle place/bounding-box definitions
    │   └── visualize_data.py      # Visualization utilities
    ├── osm_cache/                 # Cached OSM downloads
    ├── cache/                     # Internal cache for processing
    └── debug_*.geojson            # Debug outputs (failed/missed features)
```

---

## 🛠 Developer Guide

### Scripts Overview

- **`fetch_osm_data.py`**  
  Downloads raw OSM data for the configured area (roads, sidewalks, crosswalks, curb ramps, pedestrian zones, buildings).  

- **`process_data.py`**  
  Core processing pipeline:  
  - Fix road geometries (snap, merge, segment).  
  - Buffer roads into polygons.  
  - Generate crosswalk polygons & connectors.  
  - Clean sidewalks and pedestrian zones.  
  - Subtract overlaps (crosswalks from roads).  
  - Export processed GeoJSON layers.  

- **`clip.py`**  
  Provides clipping utilities to limit datasets to bounding boxes. I reccommend running this after fetching to not have to process too much data.

- **`location.py`**  
  Handles location and bounding box definitions for fetching data.  

- **`visualize_data.py`**  
  Simple visualization of GeoJSON outputs.  

### Outputs

- **Raw layers**: Direct OSM extracts (`raw_roads.geojson`, `raw_sidewalks.geojson`, etc.).  
- **Processed layers**: Cleaned and enriched datasets (`processed_accessibility.geojson`, `processed_pedestrian_network.geojson`, etc.).  
- **Debug layers**: Identify dropped or missed features (`debug_missed_crosswalks.geojson`, `debug_crosswalk_flags.geojson`).  

### Logging & Debugging

- Logs stored in `logs/` (e.g., `osmnx_YYYY-MM-DD.log`).  
- Debug GeoJSONs are written when certain steps fail (e.g., dropped roads or missed crosswalks).  

---

## 🎯 Applications

- Pedestrian accessibility and ADA compliance analysis.  
- Walkability and safety studies.  
- Urban planning and design.  
- Input for routing/simulation models.  
- Quick **GeoJSON generation** for GIS tools.  
