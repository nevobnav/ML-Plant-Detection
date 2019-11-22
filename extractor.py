#!/usr/bin/python3.6

"""
Script containing functions that extract training data from existing point
and contour shapefiles.
"""

import os

import numpy as np
from PIL import Image
import pickle
import fiona
from PIL import Image, ImageDraw

from rasters import RasterCommunicator

def create_mask(xs, ys, box_size, fill=1):
	"""Converts contour to binary mask.

	Arguments
	---------
	xs : list of length N
		X-coordinates of contour relative to box.
	ys : list of length N
		Y-coordinates of contour relative to box.
	box_size : int
		Size of box in terms of pixels.
	fill : int, optional
		Value used to fill interior of contour. Keep at default value
		of 1. This will make the masks compatible with mean squared error
		loss function while training model.

	Returns
	-------
	image : PIL Image object
		Image containing mask.
	"""

	poly = [(xs[k], ys[k]) for k in range(len(xs))]
	tmp = Image.new('L', (box_size, box_size), 0)
	ImageDraw.Draw(tmp).polygon(poly, outline=1, fill=fill)
	return tmp

def get_empty_mask(box_size):
	"""Creates an empty mask.

	Arguments
	---------
	box_size : int
		Size of box in terms of pixels.

	Returns
	-------
	image : PIL Image object
		Empty Image object.
	"""

	tmp = Image.new('L', (box_size, box_size), 0)
	return tmp

def save_data(data, targets, label):
	"""Save data to their respective target directory. 

	The RGB data will get the name RGB+label, the DEM data
	will get the name DEM+label, and the mask data will get the name
	MSK+label.

	Arguments
	---------
	data : 3-tuple
		Tuple containing (RGB_array, DEM_array, MSK_array)
	targets : 3-tuple
		Tuple containing (RGB_path, DEM_path, MSK_path)
	label : str
		Label to add to title of files.
	"""

	rgb_data, dem_data, mask = data
	dir_rgb, dir_dem, dir_msk = targets
	dem_data = (dem_data - dem_data.min())
	dem_data *= 255/dem_data.max()			# normalize height
	rgb_data = rgb_data.astype(np.uint8)
	dem_data = dem_data.astype(np.uint8)
	rgb_pil = Image.fromarray(rgb_data, 'RGB')
	dem_pil = Image.fromarray(dem_data, 'L')
	rgb_save_path = dir_rgb +'RGB_'+label+'.png'
	dem_save_path = dir_dem+'DEM_'+label+'.png'
	msk_save_path = dir_msk +'MSK_'+label+'.png'
	rgb_pil.save(rgb_save_path)
	dem_pil.save(dem_save_path)
	mask.save(msk_save_path)

def extract_background_data(background_pickle, rgb_path, dem_path, clip_path, box_size, target_dir='./', min_conf=0.95, max_count=1000):
	"""Extracts background training data from pickle file.

	If it does not yet exist, the following tree will be created:
		target_dir
		|-- RGB
		|	|-- Class 0 - Background
		|-- DEM
		|	|-- Class 0 - Background
		|-- MSK
		|	|-- Class 0 - Background
	The dictionary stored in background_pickle should have the same structure
	as produced by detector.Detector.detect(get_background=True).

	Arguments
	---------
	background_pickle : str (path object)
		Path to pickle file in which the background box locations and 
		dimensions are stored.
	rgb_path : str (path object)
		Path to input RGB tif used to generate data.
	dem_path : str (path object)
		Path to input DEM tif used to generate data.
	clip_path : str(path object)
		Path to clipped field shapefile.
	box_size : int
		Desired box size.
	target_dir : str (path object), optional
		Path in which to create tree. Default is './'
	min_conf: float (in (0.5, 1]), optional
		Minimum confidence score of object. If confidence is lower, the data
		is not saved. Default is 0.95
	max_count : int, optional
		Number of data pairs to save. Default is 1000.
	"""

	with open(background_pickle, 'rb') as p:
		data_dict = pickle.load(p)

	class_name = 'Class 0 - Background'	

	dir_rgb = target_dir+'/RGB/'+class_name+'/'
	dir_dem = target_dir+'/DEM/'+class_name+'/'
	dir_msk = target_dir+'/MSK/'+class_name+'/'

	for direc in [dir_rgb, dir_msk, dir_dem]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	label_format = os.path.basename(rgb_path).split('.')[0]+r'_{}_{}_{}'
	Rasters = RasterCommunicator(rgb_path, dem_path, clip_path)

	count = 0
	for (i,j) in data_dict:
		boxes = data_dict[(i,j)]['background_boxes']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']
		probs = data_dict[(i,j)]['background_confidence']

		for (k, box) in enumerate(boxes):
			if probs[k,0] < min_conf:
				continue

			rgb_row = box[1] + i_ad
			rgb_col = box[0] + j_ad

			rgb_data, dem_data = Rasters.get_blocks(rgb_row, rgb_col, box_size, box_size)
			mask = get_empty_mask(box_size)

			save_data(data = (rgb_data, dem_data, mask), 
					  targets = (dir_rgb, dir_dem, dir_msk),
					  label = label_format.format(rgb_row, rgb_col, k))

			count += 1
			if count >= max_count:
				print('Saved {} data-triplets of class "{}"'.format(count, class_name))
				return

def extract_data_from_shapefiles(contour_shapefile, point_shapefile, rgb_path, dem_path, clip_path, \
			box_size, class_id, class_title='', target_dir='./', min_conf=0.95, max_count=1000, filter_id=False):
	"""Extracts training data from shapefiles.

	If it does not yet exist, the following tree will be created:
		target_dir
		|-- RGB
		|	|-- Class {class_id} - {class_title}
		|-- DEM
		|	|-- Class {class_id} - {class_title}
		|-- MSK
		|	|-- Class {class_id} - {class_title}
	We recommend that the class name is of the form "Class k - Class_Title",
	to make it compatible with the NetworkTrainer class in network.py.

	Arguments
	---------
	contour_shapefile : str (path object)
		Path to shapefile containing contours.
	point_shapefile : str (path object)
		Path to shapefile containing points.
	rgb_path : str (path object)
		Path to input RGB tif used to generate data.
	dem_path : str (path object)
		Path to input DEM tif used to generate data.
	clip_path : str(path object)
		Path to clipped field shapefile.
	box_size : int
		Desired box size.
	class_id : int (>=1)
		Identification integer of class. The 0-class is reserved for background.
	class_title : str
		Name of the class. It will be added to the folder name.
	target_dir : str (path object), optional
		Path in which to create tree. Default is './'
	min_conf: float (in (0.5, 1]), optional
		Minimum confidence score of object. If confidence is lower, the data
		is not saved. Default is 0.95
	max_count : int, optional
		Number of data pairs to save. Default is 1000.
	filter_id : bool, optional
		Whether to only save data of which the property 'class_id' coincides
		with the argument class_id. Set to True if there are multiple classes of
		crops in the field. The function needs to be called once for every class.
		If there is only one class of crops present, keep at False. In this case
		the property 'class_id' will be ignored, which is handy for older
		datasets that do not have this property.

	Raises
	------
	IndexError : if a point and contour at the same index have different names.
	"""

	class_name = 'Class {} - '.format(class_id)+class_title
	dir_rgb = target_dir+'/RGB/'+class_name+'/'
	dir_dem = target_dir+'/DEM/'+class_name+'/'
	dir_msk = target_dir+'/MSK/'+class_name+'/'

	for direc in [dir_rgb, dir_msk, dir_dem]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	label_format = os.path.basename(rgb_path).split('.')[0]+r'_{}_{}_{}'
	Rasters = RasterCommunicator(rgb_path, dem_path, clip_path)

	shp_cnts = fiona.open(contour_shapefile)
	shp_pnts = fiona.open(point_shapefile)

	count = 0
	for k in range(len(shp_pnts)):
		pnt, cnt = shp_pnts[k], shp_cnts[k]

		if pnt['properties']['confidence'] < min_conf:
			continue

		if filter_id:
			if pnt['properties']['class_id'] != class_id:
				continue

		if pnt['properties']['name'] != cnt['properties']['name']:
			raise IndexError('Point and Contour do not have the same name; datasets not of the same form.')

		pnt_coords = pnt['geometry']['coordinates']
		cnt_coords = cnt['geometry']['coordinates'][0]

		pnt_index = Rasters.rgb_index(*pnt_coords)
		cnt_index = np.array([list(Rasters.rgb_index(x,y)) for (x,y) in cnt_coords])

		relative_cnt = cnt_index - np.ones(cnt_index.shape)*(pnt_index[0]-box_size/2, pnt_index[1]-box_size/2)
		rgb_row, rgb_col = (np.round(pnt_index)-np.array([box_size/2, box_size/2])).astype(int)

		rgb_data, dem_data = Rasters.get_blocks(rgb_row, rgb_col, box_size, box_size)
		mask = create_mask(relative_cnt[:,1], relative_cnt[:,0], box_size, fill=1)

		save_data(data = (rgb_data, dem_data, mask), 
				  targets = (dir_rgb, dir_dem, dir_msk),
				  label = label_format.format(rgb_row, rgb_col, k))

		count += 1
		if count >= max_count:
			print('Saved {} data-triplets of class "{}"'.format(count, class_name))
			return

