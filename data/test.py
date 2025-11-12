import pandas as pd
import geopandas as gpd

counties_gdf = gpd.read_file('cb_2024_us_county_500k.shp')
print(counties_gdf.columns.tolist())
print(counties_gdf.head())