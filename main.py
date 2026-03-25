from structural_metrics import computemetrics 
#You need an init file so that you can call functions inside the directory. 

#List of files 
files = ["data/NR8931_10PPM_LAZ_ScotlandNationalLiDAR.laz"]
outputs = []
for file in files: 
    metric_files = computemetrics(file) 
    outputs.append(metric_files)

print (outputs)