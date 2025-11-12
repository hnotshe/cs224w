import pandas as pd
import numpy as np
import geopandas as gpd

# Download Census TIGER shapefile (run once)
# URL: https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2024_us_county_500k.zip
# Note: The geopandas.datasets module was deprecated and removed in GeoPandas 1.0
# For Natural Earth data, download from: https://www.naturalearthdata.com/downloads/

# counties_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# ^ This is deprecated; for accurate data use Census TIGER below

# Better: use Census directly
import requests
import zipfile
import os
import glob

# Use 2024 county data (500k resolution - good balance of detail and file size)
# Note: 2024 files are in the GENZ2023 directory
url = 'https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip'
zip_path = 'counties.zip'
shapefile_name = 'cb_2024_us_county_500k.shp'

# Download with requests (handles SSL better)
if not os.path.exists(zip_path):
    print("Downloading Census TIGER county shapefile...")
    response = requests.get(url, verify=True)
    response.raise_for_status()
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print("✓ Download complete")
else:
    print(f"✓ Using existing {zip_path}")

# Extract the shapefile
print("Extracting shapefile...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('.')
print("✓ Extraction complete")

# Load the shapefile (try the expected name, or find any .shp file in the zip)
if os.path.exists(shapefile_name):
    counties_gdf = gpd.read_file(shapefile_name)
else:
    # Fallback: find the .shp file that was extracted
    shp_files = glob.glob('*.shp')
    if shp_files:
        print(f"Using extracted shapefile: {shp_files[0]}")
        counties_gdf = gpd.read_file(shp_files[0])
    else:
        raise FileNotFoundError(f"Could not find {shapefile_name} or any .shp file after extraction")
counties_gdf = gpd.read_file('cb_2024_us_county_500k.shp')
print(counties_gdf.columns.tolist())
print(counties_gdf.head())
# Extract basic county info
counties = pd.DataFrame({
    'county_id': counties_gdf['GEOID'].astype(str),
    'county_name': counties_gdf['NAME'].astype(str),
    'state_fips': counties_gdf['STATEFP'].astype(str),
    'latitude': counties_gdf.geometry.centroid.y,
    'longitude': counties_gdf.geometry.centroid.x,
    'land_area_sqm': counties_gdf['ALAND'].astype(float),
})

print(f"✓ Loaded {len(counties)} US counties")
print(counties.head())

# Map FIPS to state abbreviations (includes DC and territories)
fips_to_state = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA',
    '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
    '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO',
    '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ',
    '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC',
    '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT',
    '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY',
    '60': 'AS',  # American Samoa
    '66': 'GU',  # Guam
    '69': 'MP',  # Northern Mariana Islands
    '72': 'PR',  # Puerto Rico
    '78': 'VI',  # U.S. Virgin Islands
}

counties['state_abbr'] = counties['state_fips'].map(fips_to_state)

print("✓ Added state abbreviations")
print(counties.head())

# Map electricity costs by state (cents per kWh)
electricity_cost_cents_per_kwh = {
    'AL': 15.30, 'AK': 25.36, 'AZ': 14.12, 'AR': 11.99, 'CA': 30.45,
    'CO': 14.87, 'CT': 27.04, 'DE': 14.72, 'DC': 21.98, 'FL': 13.44,
    'GA': 13.46, 'HI': 36.95, 'ID': 10.49, 'IL': 16.02, 'IN': 15.34,
    'IA': 14.45, 'KS': 13.36, 'KY': 12.75, 'LA': 11.67, 'ME': 24.77,
    'MD': 17.49, 'MA': 26.96, 'MI': 17.80, 'MN': 15.05, 'MS': 13.05,
    'MO': 13.91, 'MT': 13.43, 'NE': 11.36, 'NV': 11.22, 'NH': 22.45,
    'NJ': 21.83, 'NM': 14.23, 'NY': 24.54, 'NC': 12.30, 'ND': 10.23,
    'OH': 14.55, 'OK': 12.15, 'OR': 13.52, 'PA': 16.33, 'RI': 24.42,
    'SC': 12.86, 'SD': 12.46, 'TN': 13.06, 'TX': 12.27, 'UT': 12.26,
    'VT': 20.90, 'VA': 12.85, 'WA': 12.75, 'WV': 13.62, 'WI': 16.10,
    'WY': 12.06,
    # Territories (using average or placeholder - update with actual data if available)
    'AS': 20.00,  # American Samoa (placeholder)
    'GU': 20.00,  # Guam (placeholder)
    'MP': 20.00,  # Northern Mariana Islands (placeholder)
    'PR': 20.00,  # Puerto Rico (placeholder)
    'VI': 20.00,  # U.S. Virgin Islands (placeholder)
}

# Add electricity cost to counties dataframe
counties['electricity_cost_cents_per_kwh'] = counties['state_abbr'].map(electricity_cost_cents_per_kwh)

print("✓ Added electricity costs")
print(counties[['county_name', 'state_abbr', 'electricity_cost_cents_per_kwh']].head(20))

# Save processed counties data
output_dir = 'processed'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'counties.csv')
counties.to_csv(output_csv, index=False)
print(f"✓ Saved processed counties data to {output_csv}")

# Also save the full GeoDataFrame with geometry (useful for spatial operations)
output_geojson = os.path.join(output_dir, 'counties.geojson')
counties_gdf.to_file(output_geojson, driver='GeoJSON')
print(f"✓ Saved counties GeoDataFrame with geometry to {output_geojson}")