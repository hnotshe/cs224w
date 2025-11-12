import osmnx as ox
from pathlib import Path
import pandas as pd
import sys

# Get project root (parent of data directory)
project_root = Path(__file__).parent.parent
output_path = project_root / "data" / "raw" / "osm_datacenters.csv"

# Allow testing with a smaller area via command line argument
# Usage: python data/test.py [place_name]
# Default: "United States" (can be slow, use a city/state for testing)
place_name = sys.argv[1] if len(sys.argv) > 1 else "United States"

try:
    # Fetch data centers from OpenStreetMap
    print(f"Fetching data center locations from OpenStreetMap for: {place_name}")
    print("This may take a while for large areas...")
    
    gdf = ox.features_from_place(place_name, tags={"telecom": "data_center"})
    
    if len(gdf) == 0:
        print(f"Warning: No data centers found for {place_name}")
        sys.exit(0)
    
    print(f"Found {len(gdf)} data center(s)")
    
    # Select relevant columns and convert geometry to WKT for CSV compatibility
    gdf = gdf[["geometry", "name"]].copy()
    # Convert geometry column to WKT strings for CSV export
    # Use set_geometry(None) to convert GeoDataFrame to DataFrame, then convert geometry
    geometry_wkt = gdf["geometry"].apply(lambda x: x.wkt if x is not None and not pd.isna(x) else None)
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))
    df["geometry"] = geometry_wkt
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} data centers to {output_path}")
    
except KeyboardInterrupt:
    print("\nQuery interrupted by user. Try a smaller area (e.g., 'San Francisco, California')")
    sys.exit(1)
except Exception as e:
    error_msg = str(e)
    print(f"Error: {error_msg}")
    
    if "No matching features" in error_msg:
        print("\nNo data centers found with tag 'amenity=data_center'.")
        print("Data centers may be tagged differently in OpenStreetMap.")
        print("You might want to try:")
        print("  - Different tags (e.g., 'office=datacenter', 'landuse=industrial')")
        print("  - A larger area (e.g., a state or country)")
        print("  - Check OSM tag documentation: https://wiki.openstreetmap.org/wiki/Tag:amenity%3Ddata_center")
    else:
        print("\nTip: For testing, try a smaller area:")
        print("  python data/test.py 'San Francisco, California'")
        print("  python data/test.py 'New York, New York'")
    sys.exit(1)
