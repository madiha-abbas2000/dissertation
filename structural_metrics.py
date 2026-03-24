# =============================================================
# Chapter 1 — Structural Metric Computation
# compute_metrics.py
#
# PURPOSE: Takes a ground-classified LAZ point cloud and
# produces 5 metric rasters at 10m resolution:
#   MCH    — Maximum Canopy Height
#   CHV    — Canopy Height Variation
#   FHD    — Foliage Height Diversity
#   VCI    — Vertical Complexity Index (replaces VDR)
#   Rumple — Rumple Index / proper surface rugosity
#
# LITERATURE:
#   MCH, CHV → Lefsky et al. (2002), Loke & Chisholm (2022)
#   FHD      → MacArthur & Wilson (1967), Bouvier et al. (2015)
#   VCI      → van Ewijk et al. (2011), Shokirov et al. (2023)
#   Rumple   → Jenness (2004), Parker et al. (2004)
# =============================================================

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import entropy
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

# ---------------------------------------------------------------
# STEP 1 — Load the LAZ file
# ---------------------------------------------------------------
# laspy.read() loads the entire point cloud into memory.
# las.x, las.y, las.z are numpy arrays of coordinates in
# British National Grid (EPSG:27700) — units are metres.

print("Loading LAZ file...")
las = laspy.read ("/home/s2695955/diss/data/NR8931_10PPM_LAZ_ScotlandNationalLiDAR.laz")

x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)  # This is ABSOLUTE elevation, not height above ground

# ---------------------------------------------------------------
# STEP 2 — Height normalisation using ground-classified points
# ---------------------------------------------------------------
# LiDAR files store raw elevation (e.g. 154m above sea level).
# For vegetation metrics we need HEIGHT ABOVE GROUND, not elevation.
# The file already has ground points classified as class 2
# (we confirmed this: np.unique showed [1,2,3,4,5,6]).
#
# Method:
#  a) Extract all ground points (class 2)
#  b) Build a digital terrain model (DTM) by interpolating
#     ground elevation across the tile using griddata
#  c) For every point, look up the ground elevation directly
#     below it and subtract → height above ground

print("Extracting ground points (classification = 2)...")
ground_mask = (las.classification == 2)
x_gnd = x[ground_mask]
y_gnd = y[ground_mask]
z_gnd = z[ground_mask]

# griddata does linear interpolation between known ground points.
# We interpolate at the (x, y) position of EVERY point in the cloud.
# This gives us the ground elevation directly beneath each point.
print("Interpolating ground surface beneath all points...")
z_ground_at_points = griddata(
    (x_gnd, y_gnd),   # known ground point locations
    z_gnd,             # known ground elevations
    (x, y),            # locations we want to interpolate at
    method='linear'    # linear interpolation between 3 nearest ground points
)

# Subtract ground from raw elevation to get normalised height
z_norm = z - z_ground_at_points

# Remove points where interpolation failed (NaN, outside convex hull)
# or where height is nonsensical (below ground, or impossibly tall >50m)
valid = (~np.isnan(z_norm)) & (z_norm >= 0) & (z_norm < 50)
x, y, z_norm = x[valid], y[valid], z_norm[valid]
print(f"  Retained {valid.sum():,} of {len(valid):,} points after normalisation")

# ---------------------------------------------------------------
# STEP 3 — Set up the 10m raster grid
# ---------------------------------------------------------------
# We divide the tile into 10m × 10m cells and assign
# every point to a cell by integer division.
# The grid is anchored at the tile's minimum x and maximum y
# (top-left corner in British National Grid convention).

res = 10  # 10 metre resolution

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()

cols = int((x_max - x_min) / res) + 1  # number of columns (east-west)
rows = int((y_max - y_min) / res) + 1  # number of rows (north-south)
print(f"Grid: {rows} rows × {cols} cols ({rows*res:.0f}m × {cols*res:.0f}m)")

# Column index = how far east the point is from the left edge
col_idx = ((x - x_min) / res).astype(int)

# Row index = how far south the point is from the top edge
# (y_max - y) because rasters count rows from the top downward)
row_idx = ((y_max - y) / res).astype(int)

# Safety: clip any edge-case indices to grid bounds
col_idx = np.clip(col_idx, 0, cols - 1)
row_idx = np.clip(row_idx, 0, rows - 1)

# ---------------------------------------------------------------
# STEP 4 — Initialise empty output rasters (filled with NaN)
# ---------------------------------------------------------------
# NaN (Not a Number) is our nodata value — cells with fewer
# than 5 points will stay NaN and be excluded from analysis.

MCH    = np.full((rows, cols), np.nan)  # Maximum Canopy Height
CHV    = np.full((rows, cols), np.nan)  # Canopy Height Variation
FHD    = np.full((rows, cols), np.nan)  # Foliage Height Diversity
VCI    = np.full((rows, cols), np.nan)  # Vertical Complexity Index
RUMPLE = np.full((rows, cols), np.nan)  # Rumple Index

# ---------------------------------------------------------------
# STEP 5 — Compute metrics cell by cell
# ---------------------------------------------------------------
# For each 10m cell, we collect all the normalised heights
# that fall within it, then compute all 5 metrics.
# The nested loop is slow but transparent — you can see
# exactly what is being calculated for each cell.

print("Computing metrics (this takes ~10 min for a 1km² tile)...")

# Pre-build a flat index to speed up point lookup
# (instead of comparing two arrays per cell, we use a single combined index)
flat_idx = row_idx * cols + col_idx  # unique integer per cell

for r in range(rows):
    for c in range(cols):

        # Find all points in this cell
        mask = (flat_idx == (r * cols + c))
        pts = z_norm[mask]

        # Skip cells with too few points to be meaningful
        if len(pts) < 5:
            continue

        h_max = pts.max()

        # Skip cells that are essentially bare ground (no vegetation)
        if h_max < 0.5:
            continue

        # --- MCH: Maximum Canopy Height ---
        # Simply the tallest normalised point in the cell.
        # Lefsky et al. (2002) use the 99th percentile to reduce
        # sensitivity to outliers; we use max for simplicity here.
        # Change to np.percentile(pts, 99) if outliers are a concern.
        MCH[r, c] = h_max

        # --- CHV: Canopy Height Variation ---
        # Standard deviation of all normalised heights in the cell.
        # High CHV = structurally diverse (mixed heights).
        # Low CHV  = uniform (e.g. even-aged plantation).
        # Loke & Chisholm (2022), Vepakomma et al. (2008).
        CHV[r, c] = pts.std()

        # --- FHD: Foliage Height Diversity ---
        # Shannon entropy of the VERTICAL DISTRIBUTION of returns.
        # We divide the height range into 10 equal bins, count
        # how many points fall in each bin, then apply Shannon's formula:
        #   FHD = -Σ(p_i × log(p_i))
        # where p_i is the fraction of points in bin i.
        # High FHD = returns spread across many layers = complex canopy.
        # Low FHD  = returns concentrated in 1-2 layers = simple structure.
        # MacArthur & Wilson (1967); adapted for LiDAR by Bouvier et al. (2015).
        bins = np.linspace(0, h_max + 0.01, 11)        # 10 equal height bins
        counts, _ = np.histogram(pts, bins=bins)         # count points per bin
        probs = counts / counts.sum()                    # convert to proportions
        probs = probs[probs > 0]                         # remove zero bins (log(0) undefined)
        FHD[r, c] = entropy(probs)                       # scipy entropy = -Σ p log(p)

        # --- VCI: Vertical Complexity Index ---
        # REPLACES VDR (Vertical Distribution Ratio).
        # VCI is normalised Shannon entropy across fixed height bins:
        #
        #   VCI = [Σ(p_i × ln(p_i))] / ln(number of bins)
        #
        # Unlike FHD which uses bins relative to local max height,
        # VCI uses FIXED absolute height bins (from van Ewijk et al. 2011).
        # This makes it directly comparable across cells regardless of tree height.
        # VCI = 1.0 means perfectly even distribution across all bins.
        # VCI → 0 means all returns concentrated in one bin.
        #
        # We use the height cuts from the lidarSHM paper (Shokirov et al. 2023):
        # bins at 0, 2, 5, 10, 15, 35m — matched to vegetation strata.
        shannon_cuts = np.array([0, 2, 5, 10, 15, 35])  # height bin edges in metres
        n_bins = len(shannon_cuts) - 1                    # = 5 bins
        vci_counts, _ = np.histogram(pts, bins=shannon_cuts)
        vci_total = vci_counts.sum()
        if vci_total > 0:
            vci_probs = vci_counts / vci_total
            vci_probs = vci_probs[vci_probs > 0]  # drop empty bins
            raw_entropy = -np.sum(vci_probs * np.log(vci_probs))
            VCI[r, c] = raw_entropy / np.log(n_bins)  # normalise by log(n_bins)

        # --- RUMPLE INDEX: proper surface rugosity ---
        # REPLACES the simple std/max proxy from the earlier script.
        # The Rumple Index is defined as:
        #
        #   Rumple = (3D canopy surface area) / (2D ground area)
        #
        # A flat canopy surface → Rumple = 1.0.
        # A very rough, complex canopy → Rumple >> 1.0.
        #
        # METHOD: We treat the top-of-canopy as a surface.
        # For each point, we use its (x, y) position and z_norm height.
        # We triangulate the point cloud surface using Delaunay triangulation
        # and sum the 3D area of all triangles.
        # Then we divide by the 2D footprint area of the cell (res × res).
        #
        # Jenness (2004); Parker et al. (2004).
        #
        # Note: for a proper canopy surface model, we'd want only
        # the FIRST RETURN points here. For now we use all points.
        # If las.return_number is available, filter to return_number == 1.
        pts_x = x[mask]
        pts_y = y[mask]
        # Build 2D Delaunay triangulation using (x, y) coordinates
        if len(pts_x) >= 3:  # need at least 3 points to make a triangle
            try:
                tri = Delaunay(np.column_stack([pts_x, pts_y]))
                total_3d_area = 0.0
                for simplex in tri.simplices:
                    # Each simplex is a triangle defined by 3 point indices
                    # Get the 3D coordinates of each vertex
                    v0 = np.array([pts_x[simplex[0]], pts_y[simplex[0]], pts[simplex[0]]])
                    v1 = np.array([pts_x[simplex[1]], pts_y[simplex[1]], pts[simplex[1]]])
                    v2 = np.array([pts_x[simplex[2]], pts_y[simplex[2]], pts[simplex[2]]])
                    # Area of triangle = 0.5 × |cross product of two edge vectors|
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    cross = np.cross(edge1, edge2)
                    triangle_area = 0.5 * np.linalg.norm(cross)
                    total_3d_area += triangle_area
                # Divide 3D surface area by 2D footprint of the cell
                ground_area = res * res  # = 100 m² for a 10m cell
                RUMPLE[r, c] = total_3d_area / ground_area
            except Exception:
                pass  # if triangulation fails (e.g. collinear points), leave as NaN

    # Print progress every 10 rows so you know it's still running
    if r % 10 == 0:
        pct = 100 * r / rows
        print(f"  Row {r}/{rows} ({pct:.0f}%)...")

# ---------------------------------------------------------------
# STEP 6 — Save all 5 rasters as GeoTIFF files
# ---------------------------------------------------------------
# from_origin() defines the affine transform:
#   top-left corner = (x_min, y_max)
#   pixel size = res (10m)
# EPSG:27700 = British National Grid (standard for Scottish data)

print("Saving rasters...")
transform = from_origin(x_min, y_max, res, res)
crs = "EPSG:27700"

for name, arr in [("MCH", MCH), ("CHV", CHV), ("FHD", FHD),
                  ("VCI", VCI), ("RUMPLE", RUMPLE)]:
    out_path = f"/home/s2695955/diss/outputs/{name}_10m.tif"
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=rows, width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"  Saved {out_path}")

print("\nAll done! Check your outputs/ folder.")
print("Open any .tif in QGIS to inspect visually before proceeding.")