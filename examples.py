#!/usr/bin/python3.6

"""
Example usage of the detector, settings and network modules. 
"""

import detector
import settings
import network
import extractor

def detect_example_linux():
	"""Example of broccoli crop detection workflow on linux"""
	name = 'c01_verdonk-Wever oost-201907240707'
	GR = True
	rgb_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"

	Settings = settings.BroccoliUnifiedSettings(model_path='/home/duncan/Documents/VanBoven/Code/Git Folder/Unified CNNs/Broccoli v4')
	D = detector.Detector(rgb_path, dem_path, clp_path, Settings)

	D.detect(max_count=5, get_background=True)
	D.remove_duplicate_crops()
	D.write_shapefiles('./Test Count')
	D.save_background_to_pickle('./Test Count')

def detect_example_windows():
	"""Example of broccoli crop detection workflow on windows"""
	rgb_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724-GR.tif"
	dem_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724_DEM-GR.tif"
	clp_path = r"C:\\Users\\VanBoven\\Documents\\DL Plant Count\\ML-Plant-Detection\\Field Shapefiles\\c01_verdonk-Wever west-201907240724-GR_FIELD.shp"

	Settings = settings.BroccoliUnifiedSettings()
	D = detector.Detector(rgb_path, dem_path, clp_path, Settings, platform='windows')

	D.detect(max_count=10)
	D.remove_duplicate_crops()
	D.write_shapefiles()

def network_train_example():
	"""Example of training a broccoli network"""
	data_path = '/home/duncan/Documents/VanBoven/DL Datasets/Unified Broccoli (NEW FORMAT)'

	Trainer = network.NetworkTrainer(2)
	Trainer.compile()
	Trainer.set_training_data(data_path)
	Trainer.train(1,1)
	Trainer.save('./Test Network')

def extraction_example():
	name = 'c01_verdonk-Wever oost-201907240707'
	GR = True
	rgb_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"
	input_contours = r"./Test Count/CONTOURS.shp"
	input_centers  = r"./Test Count/POINTS.shp"
	bg_pickle = r"./Test Count/BACKGROUND_DATA.pickle"

	extractor.extract_data_from_shapefiles(contour_shapefile = input_contours, 
								 point_shapefile = input_centers, 
								 rgb_path = rgb_path, 
								 dem_path = dem_path, 
								 clip_path = clp_path, 
								 box_size = 50, 
								 class_id = 1,
								 class_title = 'Broccoli', 
								 target_dir = './Test Automatic Data', 
								 max_count = 10, 
								 min_conf = 0.6,
								 filter_id = False)

	extractor.extract_background_data(background_pickle = bg_pickle, 
							rgb_path = rgb_path, 
							dem_path = dem_path, 
							clip_path = clp_path, 
							box_size = 50, 
							target_dir = './Test Automatic Data', 
							max_count = 10, 
							min_conf = 0.6)

if __name__ == "__main__":
	detect_example_linux()
	# detect_example_windows()
	# network_train_example()
	# extraction_example()