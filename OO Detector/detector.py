#!/usr/bin/python3.6

"""
Script containing the Detector class, which implements the crop 
detection algorithm.
"""

import os
import pickle

import cv2
import tensorflow.keras.models as keras_models
import rasterio
import shapely.geometry as geometry
import fiona
from fiona.crs import from_epsg
import numpy as np

import processing as proc
from raster_functions import RasterCommunicator
import settings

__author__ = "Duncan den Bakker"

def load_network(path, platform):
	"""Loads a neural network.

	If platform=='linux', the model is loaded from a .h5 file using 
	keras.models.load_model(path). If platform=='windows', the model
	architecture is first loaded from a .json, achter which the 
	weights are loaded separately.

	Arguments
	---------
	path : str
		Path to network. Should not have a suffix, so to load 
		./network.h5, set path='./network'.
	platform : str
		OS, either linux or windows.
	"""

	if platform=='linux':
		network = keras_models.load_model(path+'.h5')

	elif platform == 'windows':
		from keras.utils import CustomObjectScope
		from keras.initializers import glorot_uniform
		with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
			with open(path+'.json', 'r') as f:
				json_string = f.read()
				network = keras_models.model_from_json(json_string)
				network.load_weights(path+'_weights.h5')

	return network

def get_input_sizes(network):
	"""Returns the (spatial) sizes of the input layers of a network.

	Arguments
	---------
	network : keras model
		Keras model of which the input shapes need to be known.

	Returns
	-------
	shapes : tuple
		Tuple containing as many 2-tuples as there are input layers
		in the network.
	"""

	shapes = []
	for input_layer in network.input:
		shape = input_layer.shape.as_list()
		shapes.append((shape[1], shape[2]))
	return tuple(shapes)

def get_num_classes(network):
	"""Returns the number of classes the network can detect.

	Arguments
	---------
	network : keras model
		Keras model of which the number of classes it can detect
		is wanted.

	Returns:
	num_classes : int
		Number of different classes the network can detect. If the
		network is a binary detection network (crop vs. no crop), it
		will return 2.
	"""

	for output in network.output:
		if len(output.shape) == 2:
			return output.shape[-1]


class Detector(object):
	"""
	Object used to run the crop detection algorithm.

	Attributes
	----------
	rgb_path : str (path object)
		Path to input RGB tif.
	dem_path : str (path object)
		Path to input DEM tif.
	clip_path : str(path object)
		Path to clipped field shapefile.
	Settings : settings.DetectionSettings
		Object containing all parameters.
	platform : str
		Either 'linux' or 'windows', necessary because keras models
		are loaded differently in windows.
	network : keras model
		Model used for classification and masking. It is loaded
		from the path specified in Settings.
	rgb_input_shape : 2-tuple
		Tuple representing the spatial shape of rgb input layer.
	dem_input_shape : 2-tuple
		Tuple representing the spatial shape of dem input layer.
	num_classes : int (>=2)
		Number of different classes that can be detected by the network.
		If equal to 2, the model is binary.
	Rasters : raster_functions.RasterCommunicator
		Object containing functions to communicate between RGB and
		DEM tifs.
	detected_crops : dict
		Dictionary in which detection results are stored. It is initialized
		by an empty dictionary. To fill it, run detection algorithm. After 
		this is done, it will contain keys of the form (i,j), representing
		the block. Each corresponding value is a dictionary containing the
		blocks geometry and crops for each class id.
	bg_dict : dict
		Dictionary in which background box locations are stored, if
		detect() is ran with get_background=True.

	Methods
	-------
	create_boxes(rgb_coords, rgb_size)
		Creates two arrays containing RGB and DEM boxes.
	fill_data_tensors(rgb_block, dem_block, rgb_boxes, dem_boxes)
		Creates input data tensors from rgb and dem blocks and their
		respective boxes.
	run_on_block(rgb_block, dem_block, get_background=False)
		Applies detection algorithm to a part of input tifs.
	detect(self, max_count=np.infty, get_background=False)
		Detection algorithm applied to whole input tif.
	remove_overlaping_crops()
		Removes crops that have been detected multiple times in 
		overlap regions.
	write_points(output_folder='./', filter_edge=True)
		Writes only crop centroids to shapefile.
	write_shapefiles(output_folder='./', filter_edges=True, get_background=False)
		Writes crop contours, centroids and block lines to shapefiles.
	save_to_pickle(output_folder='./')
		Saves self.detected_crops to pickle file.
	save_background_to_pickle(output_folder='./')
		Saves self.bg_dict to pickle file.
	"""

	def __init__(self, rgb_path, dem_path, clip_path, Settings, platform='linux'):
		"""
		Arguments
		---------
		rgb_path : str (path object)
			Path to input RGB tif.
		dem_path : str (path object)
			Path to input DEM tif.
		clip_path : str(path object)
			Path to clipped field shapefile.
		Settings : settings.DetectionSettings
			Object containing all parameters.
		platform : str, optional
			Either 'linux' or 'windows', necessary because keras models
			are loaded differently in windows. Default is 'linux'.
		"""

		self.rgb_path = rgb_path
		self.dem_path = dem_path
		self.clip_path = clip_path
		self.Settings = Settings
		self.platform = platform

		self.network = load_network(self.Settings.model_path, platform)
		self.rgb_input_shape, self.dem_input_shape = get_input_sizes(self.network)
		self.num_classes = get_num_classes(self.network)

		self.Rasters = RasterCommunicator(self.rgb_path, self.dem_path, self.clip_path)

		self.detected_crops = {}
		self.bg_dict = {}

	def create_boxes(self, rgb_coords, rgb_size):
		"""Creates arrays containing relative to RGB and DEM.

		Arguments
		---------
		rgb_coords : (N,2) numpy array
			Array containing the row, column index in RGB tif of each box.
		rgb_size : int
			Box size.

		Returns
		-------
		rgb_boxes : (N,4) numpy array
			Array containing each RGB box location and dimension in the form
			(i_top, j_left, height, width).
		dem_boxes : (N,4) numpy array
			Array containing each DEM box location and dimension.
		"""

		dem_size = int(np.round(self.Rasters.scale_factor*rgb_size))
		dem_coords = np.round(self.Rasters.scale_factor*rgb_coords).astype(int) 									# convert to DEM row/col
		rgb_boxes = np.zeros((rgb_coords.shape[0], 4), dtype=int)
		dem_boxes = np.zeros((dem_coords.shape[0], 4), dtype=int)
		rgb_boxes[:,0] = rgb_coords[:,1] - rgb_size//2
		rgb_boxes[:,1] = rgb_coords[:,0] - rgb_size//2
		rgb_boxes[:,2] = rgb_size
		rgb_boxes[:,3] = rgb_size
		dem_boxes[:,0] = dem_coords[:,1] - dem_size//2
		dem_boxes[:,1] = dem_coords[:,0] - dem_size//2
		dem_boxes[:,2] = dem_size
		dem_boxes[:,3] = dem_size
		return rgb_boxes, dem_boxes

	def fill_data_tensors(self, rgb_block, dem_block, rgb_boxes, dem_boxes):
		"""Initializes two RGB and DEM input tensors.

		Arguments
		---------
		rgb_block : (?,?,3) numpy array
			RGB block data.
		dem_block : (?,?) numpy array
			DEM block data.
		rgb_boxes : (N,4) numpy array
			RGB box locations and dimensions.
		dem_boxes : (N,4) numpy array
			DEM box locations and dimensions.

		Returns
		-------
		rgb_input_tensor : (N,?,?,3) numpy array
			Tensor containing RGB data that should be fed through network.
		dem_input_tensor : (N,?,?,1) numpy array
			Tensor containing DEM data that should be fed through network.
		"""

		num_candidates = rgb_boxes.shape[0]
		rgb_input_tensor = np.zeros((num_candidates, *self.rgb_input_shape, 3), dtype=np.uint8)		# initialize tensors
		dem_input_tensor = np.zeros((num_candidates, *self.dem_input_shape, 1), dtype=np.uint8)
		for i in range(num_candidates):
			j_rgb, i_rgb, cols_rgb, rows_rgb = rgb_boxes[i,:]
			j_dem, i_dem, cols_dem, rows_dem = dem_boxes[i,:]
			rgb_box_data = np.copy(rgb_block[i_rgb : i_rgb+rows_rgb, j_rgb : j_rgb+cols_rgb, :]).astype(np.uint8)
			dem_box_data = np.copy(dem_block[i_dem : i_dem+rows_dem, j_dem : j_dem+cols_dem]).astype(np.uint8)
			rgb_box_data = cv2.resize(rgb_box_data, self.rgb_input_shape)
			dem_box_data = cv2.resize(dem_box_data, self.dem_input_shape)
			rgb_input_tensor[i,:,:,:] = rgb_box_data
			dem_input_tensor[i,:,:,0] = dem_box_data
		return rgb_input_tensor, dem_input_tensor

	def run_on_block(self, rgb_block, dem_block, get_background=False):
		"""Run detection algorithm on RGB block and its corresponding DEM block.

		A short summary of the detection algorithm:
		* First generate RoI's, and put boxes of a fixed size at these locations.
		* Feed the data in the boxes through a combined classification and 
			masking network.
		* Sort the results into crops and background.
		* Apply post-processing to crop results, like non-max-suppression to
			discard overlapping boxes, and clean up masks.
		* Convert masks to contours and centroids.

		Arguments
		---------
		rgb_block : (?,?,3) block
			RGB data block.
		dem_block : (?,?) block
			DEM data block corresponding to RGB block.
		get_background : bool, optional
			If set to True, all boxes that contain background are stored
			and returned

		Returns
		-------
		crop_output : list of length 4
			List containing contours, centroids, boxes and confidence scores
		bg_output : list of length 2
			List containing background boxes and confidence scores. If 
			get_background==False, this is a list containing two empty lists.	
		"""

		if rgb_block.mean() <= 1e-6:
			raise IndexError('This block contains no data.')

		rgb_coords = proc.green_hotspots(rgb_block, sigma=self.Settings.sigma, padding=self.Settings.box_size)							# run region proposer
		rgb_boxes, dem_boxes = self.create_boxes(rgb_coords, self.Settings.box_size)
		rgb_input_tensor, dem_input_tensor = self.fill_data_tensors(rgb_block, dem_block, rgb_boxes, dem_boxes)

		predictions, masks = self.network.predict([rgb_input_tensor, dem_input_tensor], verbose=1)							# run classification model
		masks = masks[...,0]

		output = dict()

		for class_idx in range(1, self.num_classes):
			cls_idxs = proc.get_class_idxs(predictions, class_idx)
			cls_boxes, [cls_confidence, cls_masks] = rgb_boxes[cls_idxs], [predictions[cls_idxs], masks[cls_idxs]]
			cls_boxes, [cls_confidence, cls_masks] = proc.non_max_suppression(cls_boxes, other=[cls_confidence, cls_masks], t=self.Settings.overlap_threshold)

			cls_masks = proc.get_hard_masks(cls_masks)
			cls_masks, cls_boxes, [cls_confidence] = proc.discard_empty(cls_masks, cls_boxes, other=[cls_confidence], t=self.Settings.crop_size_threshold)

			cls_contours  = proc.find_contours(cls_boxes, cls_masks)
			cls_centroids = proc.find_centroids(cls_boxes, cls_masks)

			output[class_idx] = {'contours'   : cls_contours,
								 'centroids'  : cls_centroids,
								 'boxes'	  : cls_boxes,
								 'confidence' : cls_confidence}

		if get_background:
			background_boxes, background_confidence = proc.get_class(rgb_boxes, predictions, 0)
			return output, [background_boxes, background_confidence]
		else:
			return output, [[],[]]

	def detect(self, max_count=np.infty, get_background=False):
		"""Apply detection algorithm to entire input file.

		The RGB tif is divided into blocks of size self.Settings.block_size.
		The detection algorithm is applied to each of these blocks. Overlap is
		added to prevent missing crops near block boundaries. This will lead 
		to some crops being detected twice in neighbouring blocks. These 
		duplicates must still be discarded using the method 
		remove_overlapping_crops().

		This method has no return value. The results of the detection algorithm
		are stored in the dictionary attribute detected_crops.

		Arguments
		---------
		max_count : int, optional
			For debugging purposes; only apply the detection algorithm to at
			most max_count blocks. Default is np.infty, in which case all blocks
			are included.
		get_background : bool, optional
			Whether to store the boxes containing background. Default is False.
		"""

		valid_blocks = self.Rasters.get_field_blocks(self.Settings.block_size, self.Settings.block_overlap, max_count=max_count)
		detected_crops = dict()
		bg_dict = dict()

		for (i,j) in valid_blocks:
			block = valid_blocks[(i,j)]
			rgb_block, dem_block = self.Rasters.get_blocks(*block)

			try:
				output, bg_output = self.run_on_block(rgb_block, dem_block, get_background=get_background)
				detected_crops[(i,j)] = {'block':block}

				for class_idx in range(1, self.num_classes):
					detected_crops[(i,j)][class_idx] = output[class_idx]
					print('Added {} crops with class id {} to block ({},{})\n'.format(output[class_idx]['centroids'].shape[0], class_idx, i, j))

				if get_background:
					background_boxes, background_confidence = bg_output
					bg_dict[(i,j)] = {'background_boxes'	  : background_boxes,
									  'background_confidence' :	background_confidence,
									  'block'				  : block}

			except Exception as e:
				print('Discarded all crops somewhere in pipeline while processing block ({},{})'.format(i,j))
				print('Exception raised: "{}"\n'.format(e))

		self.detected_crops = detected_crops
		self.bg_dict = bg_dict

	def remove_overlapping_crops(self):
		"""Removes duplicates from overlapping regions.

		This method calls the function process_overlap from the module 
		processing.py. The method has no return value, results are stored
		in the dictionary attribute detected_crops.
		"""

		self.detected_crops = proc.process_overlap(self.detected_crops, self.num_classes, self.Settings.block_overlap, self.Settings.centroid_distance)

	def write_points(self, output_folder='./', filter_edges=True):
		"""Writes detected centroids to a shapefile POINTS.shp.

		Converts every centroid in each block to lon-lat coordinates, and
		writes it to a shapefile. This method has no return value.

		Arguments
		---------
		output_folder : str (path object), optional
			Folder in which to store the resulting shapefile POINTS.shp.
			If folder does not exist, it will be created. Default is './', 
			which is the current directory.
		filter_edges : bool, optional
			Whether to remove crops which intersect the clipped field edge,
			default is True. If the clipped field edge is not too close to 
			any crops, we recommend keeping it at the default.
		"""

		if len(self.detected_crops) == 0:
			raise IndexError('No crops have been detected (yet).\n\
			 To run detection algorithm, use the method detect().')

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)

		clip_polygons = self.Rasters.get_clip_polygons()
		schema_pnt   = { 'geometry':'Point',  'properties':{'name':'str' , 'class_id':'int', 'confidence':'float'}}
		
		with fiona.collection(output_folder+'POINTS.shp', "w", "ESRI Shapefile", schema_pnt, crs=from_epsg(4326)) as output_pnt:
			for (i,j) in self.detected_crops:
				(i_ad, j_ad, height, width) = self.detected_crops[(i,j)]['block']

				for class_idx in range(1, self.num_classes):
					contours   = self.detected_crops[(i,j)][class_idx]['contours']
					centroids  = self.detected_crops[(i,j)][class_idx]['centroids']
					confidence = self.detected_crops[(i,j)][class_idx]['confidence']

					count = 0
					for (k, cnt) in enumerate(contours):							# write contours
						xs, ys = cnt[:,1] + j_ad, cnt[:,0] + i_ad
						centroid = (centroids[k,0] + j_ad, centroids[k,1] + i_ad)
						transformed_contour  = geometry.Polygon([self.Rasters.transform*(xs[l], ys[l]) for l in range(len(xs))])
						transformed_centroid = geometry.Point(self.Rasters.transform*centroid)
						try:
							if transformed_contour.difference(clip_polygons).is_empty or not filter_edges:
								output_pnt.write({'properties': { 'name' : '({},{}): {}'.format(i, j, k), 
																  'class_id' : class_idx,
																  'confidence' : float(max(confidence[k]))},
								            	  'geometry': geometry.mapping(transformed_centroid)})
								count += 1
							else:
								print('Crop ({},{}):{} intersects clipped field edge'.format(i,j,k))
						except:
							print('Contour ({},{}):{} invalid'.format(i,j,k))
					print('{} points written to block ({},{})'.format(count,i,j))

		print('\nFinished!')

	def write_shapefiles(self, output_folder='./', filter_edges=True):
		"""Writes centroids, contours and block lines to shapefiles.

		Converts every centroid and contour in each block to lon-lat 
		coordinates, and writes them to the shapefiles POINTS.shp and
		CONTOURS.shp respectively. Also saves the boundaries of each
		block to the shapefile BLOCK_LINES.shp. This method has no 
		return value.

		Arguments
		---------
		output_folder : str (path object), optional
			Folder in which to store the resulting shapefiles. If it does not
			exist, it will be created. Default is './', which is the current 
			directory.
		filter_edges : bool, optional
			Whether to remove crops which intersect the clipped field edge,
			default is True. If the clipped field edge is not too close to 
			any crops, we recommend keeping it at the default.
		"""

		if len(self.detected_crops) == 0:
			raise IndexError('No crops have been detected (yet).\n\
			 To run detection algorithm, use the method detect().')

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)

		clip_polygons = self.Rasters.get_clip_polygons()

		schema_lines = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }
		schema_pnt   = { 'geometry': 'Point',   'properties': { 'name': 'str' , 'class_id':'int', 'confidence':'float'} }
		schema_cnt   = { 'geometry': 'Polygon', 'properties': { 'name': 'str' , 'class_id':'int', 'confidence':'float'} }

		with fiona.collection(output_folder+'/CONTOURS.shp', "w", "ESRI Shapefile", schema_cnt, crs=from_epsg(4326)) as output_cnt:					# add projection
			with fiona.collection(output_folder+'/POINTS.shp', "w", "ESRI Shapefile", schema_pnt, crs=from_epsg(4326)) as output_pnt:
				with fiona.collection(output_folder+'/BLOCK_LINES.shp', "w", "ESRI Shapefile", schema_lines, crs=from_epsg(4326)) as output_lines:
					num_crops = 0
					for (i,j) in self.detected_crops:
						(i_ad, j_ad, height, width) = self.detected_crops[(i,j)]['block']

						for class_idx in range(1, self.num_classes):
							contours   = self.detected_crops[(i,j)][class_idx]['contours']
							centroids  = self.detected_crops[(i,j)][class_idx]['centroids']
							confidence = self.detected_crops[(i,j)][class_idx]['confidence']
							count = 0
							for (k, cnt) in enumerate(contours):							# write contours
								xs, ys = cnt[:,1] + j_ad, cnt[:,0] + i_ad
								centroid = (centroids[k,0] + j_ad, centroids[k,1] + i_ad)
								transformed_contour  = geometry.Polygon([self.Rasters.transform*(xs[l], ys[l]) for l in range(len(xs))])
								transformed_centroid = geometry.Point(self.Rasters.transform*centroid)
								try:
									if transformed_contour.difference(clip_polygons).is_empty or not filter_edges:
										output_cnt.write({'properties': { 'name' : '({},{}): {}'.format(i, j, k), 
																		  'class_id' : class_idx,
																		  'confidence' : float(max(confidence[k]))},
									            		  'geometry': geometry.mapping(transformed_contour)})
										output_pnt.write({'properties': { 'name' : '({},{}): {}'.format(i, j, k), 
																		  'class_id' : class_idx,
																		  'confidence' : float(max(confidence[k]))},
										            	  'geometry': geometry.mapping(transformed_centroid)})
										count += 1
									else:
										print('Crop ({},{}):{} intersects clipped field edge'.format(i,j,k))
								except:
									print('Contour ({},{}):{} invalid'.format(i,j,k))

							num_crops += count
							print('{} crops written to block ({},{})'.format(count,i,j))

						block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
						transformed_vertices = [self.Rasters.transform*(a,b) for (b,a) in block_vertices]
						output_lines.write({'properties' : {'name': 'block ({},{})'.format(i,j)},
											'geometry' : geometry.mapping(geometry.Polygon(transformed_vertices))})

		print('\nSuccesfully saved {} crops'.format(num_crops))

	def save_to_pickle(self, output_folder='./'):
		"""Saves the attribute detected_crops to a pickle file DATA.pickle.

		The dictionary detected_crops is saved as is; its contents are not
		converted to lon-lat coordinates.
		
		Arguments
		---------
		output_folder, str (path object), optional
			Folder in which to store the resulting shapefile DATA.pickle.
			If it does not exist, it will be created. Default is './', which 
			is the current directory.
		"""

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)

		with open(output_folder+'/DATA.pickle', 'wb') as file:
			pickle.dump(self.detected_crops, file)

	def save_background_to_pickle(self, output_folder='./'):
		"""Saves the attribute bg_dict to a pickle file BG_DATA.pickle.
		
		Arguments
		---------
		output_folder, str (path object), optional
			Folder in which to store the resulting shapefile BG_DATA.pickle.
			If it does not exist, it will be created. Default is './', which 
			is the current directory.
		"""

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)

		if len(bg_dict) == 0:
			raise IndexError('No background boxes saved. Run detect(get_background==True) to generate them.')

		with open(output_folder+'/BACKGROUND_DATA.pickle', 'wb') as bg_file:
			pickle.dump(self.bg_dict, bg_file)

