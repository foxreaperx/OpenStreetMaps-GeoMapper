import folium
import geopandas as gpd
from pathlib import Path
import json

def visualize():
    outputs_dir = Path(__file__).parent.parent / "outputs"
    
    try:
        # Load processed datasets
        streets_buildings = gpd.read_file(outputs_dir / "processed_streets_buildings.geojson")
        pedestrian_network = gpd.read_file(outputs_dir / "processed_pedestrian_network.geojson")
        accessibility = gpd.read_file(outputs_dir / "processed_accessibility.geojson")
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Please run process_data.py first to generate the required files")
        return

    # Create map centered on San Francisco
    m = folium.Map(location=[37.78, -122.42], zoom_start=14, tiles='CartoDB positron')
    
    # Style dictionaries with default fallback
    styles = {
        'street': {'color': '#1f78b4', 'weight': 3, 'opacity': 0.7},
        'building': {'color': '#a6cee3', 'fill': True, 'fillOpacity': 0.3},
        'sidewalk': {'color': '#33a02c', 'weight': 2, 'opacity': 0.7},
        'crosswalk': {'color': '#e31a1c', 'weight': 3, 'opacity': 0.9},
        'curb_ramp': {'color': '#ff7f00', 'radius': 3, 'fill': True},
        'pedestrian': {'color': '#6a3d9a', 'weight': 3, 'opacity': 0.6},
        'default': {'color': '#666666', 'weight': 2, 'opacity': 0.5}  # Fallback style
    }

    # Safe style function with fallback
    def get_style(feature):
        feature_type = feature.get('properties', {}).get('type')
        return styles.get(feature_type, styles['default'])

    # Add streets and buildings
    folium.GeoJson(
        streets_buildings,
        style_function=get_style,
        name='Streets & Buildings'
    ).add_to(m)

    # Add pedestrian network
    folium.GeoJson(
        pedestrian_network,
        style_function=get_style,
        name='Pedestrian Network'
    ).add_to(m)

    # Add accessibility features
    if not accessibility.empty:
        # Separate points (curb ramps) from other geometries
        points = accessibility[accessibility.geometry.type == 'Point']
        other_geoms = accessibility[accessibility.geometry.type != 'Point']
        
        # Add point features as CircleMarkers
        for _, row in points.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=styles['curb_ramp']['radius'],
                color=styles['curb_ramp']['color'],
                fill=True,
                fill_opacity=0.7,
                popup=f"Type: {row['type']}"
            ).add_to(m)
        
        # Add other geometries as GeoJson
        if not other_geoms.empty:
            folium.GeoJson(
                other_geoms,
                style_function=get_style,
                name='Accessibility Features'
            ).add_to(m)

    # Add layer control and save
    folium.LayerControl().add_to(m)
    m.save(outputs_dir / "walkability_map.html")
    print("✅ Map saved to /outputs/walkability_map.html")

if __name__ == "__main__":
    visualize()