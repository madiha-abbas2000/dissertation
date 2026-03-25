import geopandas as gpd
import rasterio
from shapely.geometry import box

# Change this path to wherever your file is
hab = gpd.read_file("/home/s2695955/diss/data/isleofarran/isleofarran.gpkg")

# print("CRS:", hab.crs)
# print("Columns:", hab.columns.tolist())
# print("Number of features:", len(hab))
# print("\nFirst few rows:")
# print(hab.head())
# print("\nUnique habitat classes:")
# print(hab.iloc[:, 1].value_counts())  # look at second column — adjust if needed


# Get the extent of your LiDAR tile from one of your metric rasters
with rasterio.open("/home/s2695955/diss/outputs/MCH_10m.tif") as src:
    bounds = src.bounds
    lidar_crs = src.crs

# Create a bounding box from the LiDAR tile extent
lidar_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
lidar_gdf = gpd.GeoDataFrame(geometry=[lidar_box], crs=lidar_crs)

# Clip habitat map to just the tile area
hab_clipped = gpd.clip(hab, lidar_gdf)

print(f"Features in tile: {len(hab_clipped)}")
print(f"\nHabitat classes present in your tile:")
print(hab_clipped.groupby(['HABITAT_CO', 'HABITAT_NA'])['Shape_Area'].sum().sort_values(ascending=False))