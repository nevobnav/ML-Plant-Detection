#!/usr/bin/python3.6

"""
Example usage of the Object Oriented version of the crop detection
algorithm. 
"""

from detector import Detector
import settings

__author__ = "Duncan den Bakker"

# On Linux
name = 'c01_verdonk-Wever oost-201907240707'
GR = True
rgb_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
dem_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
clp_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"

Settings = settings.BroccoliUnifiedSettings()
D = Detector(rgb_path, dem_path, clp_path, Settings)

D.detect(max_count=10)
D.remove_overlapping_crops()
D.write_shapefiles()

# On Windows
