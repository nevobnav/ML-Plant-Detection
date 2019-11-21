#!/usr/bin/python3.6

"""
Example usage of the detector, settings and network modules. 
"""

import detector
import settings
import network

def detect_example_linux():
	name = 'c01_verdonk-Wever oost-201907240707'
	GR = True
	rgb_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"

	Settings = settings.BroccoliUnifiedSettings()
	D = detector.Detector(rgb_path, dem_path, clp_path, Settings)

	D.detect(max_count=10)
	D.remove_duplicate_crops()
	D.write_shapefiles('./Test Count')

def detect_example_windows():
	rgb_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724-GR.tif"
	dem_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724_DEM-GR.tif"
	clp_path = r"C:\\Users\\VanBoven\\Documents\\DL Plant Count\\ML-Plant-Detection\\Field Shapefiles\\c01_verdonk-Wever west-201907240724-GR_FIELD.shp"

	Settings = settings.BroccoliUnifiedSettings()
	D = detector.Detector(rgb_path, dem_path, clp_path, Settings, platform='windows')

	D.detect(max_count=10)
	D.remove_duplicate_crops()
	D.write_shapefiles()

def network_train_example():
	data_path = '/home/duncan/Documents/VanBoven/DL Datasets/Unified Broccoli (NEW FORMAT)'

	Trainer = network.NetworkTrainer(2)
	Trainer.compile()
	Trainer.set_training_data(data_path)
	Trainer.train(1,1)
	Trainer.save('./Test Network')

if __name__ == "__main__":
	detect_example_linux()
	# detect_example_windows()
	# network_train_example()