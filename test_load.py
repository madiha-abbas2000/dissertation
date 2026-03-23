import laspy 
import numpy as np 

las1 = laspy.read ("/home/s2695955/diss_data/NR8931_10PPM_LAZ_ScotlandNationalLiDAR.laz")
las2 = laspy.read ("/home/s2695955/diss_data/NR8932_10PPM_LAZ_ScotlandNationalLiDAR.laz")

x1,y1,z1 = las1.x, las1.y, las1.z 
x2,y2,z2 = las2.x, las2.y, las2.z 

print(f"Number of points: {len(x1):,}") # density of points 
print(f"X range: {x1.min():.1f} to {x1.max():.1f}") # confirm the crs 
print(f"Y range: {y1.min():.1f} to {y1.max():.1f}") # real terrain variation and not a sea tile 
print(f"Z range: {z1.min():.1f} to {z1.max():.1f}") # class 2 is present, so ground points are classified
print(f"Point classifications: {np.unique(las1.classification)}") # 3,4 and 5 are low/medium/high vegetation so perfect for FHD and VDR 
