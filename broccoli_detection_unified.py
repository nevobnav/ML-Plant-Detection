#!/usr/bin/python3.6
platform = 'windows'
#================================================== Imports ==================================================
import os
import cv2
# import pathlib
import pickle
if platform == 'windows':
	import keras.models as ker_models
elif platform == 'linux':
	import tensorflow.keras.models as ker_models
import rasterio
from shapely.geometry import Polygon, Point, mapping, shape, MultiPolygon
from scipy.spatial import KDTree
import fiona
from fiona.crs import from_epsg
import numpy as np

import processing as proc
import tif_functions
import settings

#================================================= Crop Type =================================================
params = settings.get_settings('broccoli_unified', box_size=50, block_size=1000)
for param in params.keys():												# load all non-string parameters
	if type(params[param]) != str:
		exec('{}={}'.format(param, params[param]))

#====================================== Get Parameters & Absolute Paths ======================================
if platform == 'linux':
	name = 'c01_verdonk-Wever oost-201907240707' #'c01_verdonk-Wever oost-201907240707' #
	GR = True
	img_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"/home/duncan/Documents/VanBoven/Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"
elif platform == 'windows':
	img_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724-GR.tif"
	dem_path = r"D:\\Old GR\\c01_verdonk-Wever west-201907240724_DEM-GR.tif"
	clp_path = r"C:\\Users\\VanBoven\\Documents\\DL Plant Count\\ML-Plant-Detection\\Field Shapefiles\\c01_verdonk-Wever west-201907240724-GR_FIELD.shp"

dem_functions 	 = tif_functions.get_functions(img_path, dem_path, clp_path)		# functions to jump between color image and heightmap
get_adj_window	 = dem_functions['get_adjusted_window']
get_block 		 = dem_functions['get_block']
transform 		 = dem_functions['transform']
dem_scale_factor = dem_functions['scale_factor']

def windows_load(path):
	name = path.split(r'\\')[-1].split('.')[0]
	with open(path, 'r') as f:
		json_string = f.read()
	new_model = ker_models.model_from_json(json_string)
	new_model.load_weights(path.split('.')[0]+'_weights.h5')
	return new_model

if platform == 'linux':
	network = ker_models.load_model(params['model_path'])
elif platform == 'windows':
	from keras.utils import CustomObjectScope
	from keras.initializers import glorot_uniform
	with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		network = windows_load(params['model_path'].replace('h5','json').replace('/', r'\\'))

#============================================ Model functions ===============================================
def get_valid_blocks(block_size, block_overlap=box_size, max_count=np.infty):
	"""For every block, determine if it is valid by checking whether it intersects with the field specified by clp_path.
	Returns a dictionary of the form (i,j) : (i_ad, j_ad, height, width), where (i,j) is the index of a valid block,
	and its value are the dimensions of the adjusted box."""
	field_shape = fiona.open(clp_path)
	field_polygons = []
	for feature in field_shape:
		poly = shape(feature['geometry'])
		field_polygons.append(poly)
	tif = rasterio.open(img_path)
	num_cols = tif.width//block_size
	num_rows = tif.height//block_size
	tif.close()

	valid_blocks = dict()
	count = 0
	for i in range(0,num_rows+1):
		for j in range(0,num_cols+1):
			i_ad, j_ad, height, width = get_adj_window(i*block_size-block_overlap, j*block_size-block_overlap,
													   block_size+2*block_overlap, block_size+2*block_overlap)
			block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
			transformed_vertices = Polygon([transform*(a,b) for (b,a) in block_vertices])
			valid = False
			for field_poly in field_polygons:
				if field_poly.intersects(transformed_vertices):
					valid = True
					break
			if valid:
				valid_blocks[(i,j)] = (i_ad, j_ad, height, width)
				count += 1

			if count >= max_count:
				break
		if count >= max_count:
			break
	print('Found {} valid blocks of a total of {}'.format(len(valid_blocks), (num_rows+1)*(num_cols+1)))
	return valid_blocks

def create_boxes(c_coords, box_size):
	"""Creates two arrays of shape (N,4) containing the locations and dimensions of the
	boxes that should be fed through the network, based on the coordinates in c_coords.
	 The first array contains boxes relative to the RGB input, the second array contains 
	 boxes relative to the DEM input."""
	h_size = int(np.round(dem_scale_factor*box_size))
	h_coords = np.round(dem_scale_factor*c_coords).astype(int) 									# convert to DEM row/col
	c_rects = np.zeros((c_coords.shape[0], 4), dtype=int)
	h_rects = np.zeros((h_coords.shape[0], 4), dtype=int)
	c_rects[:,0] = c_coords[:,1] - box_size//2
	c_rects[:,1] = c_coords[:,0] - box_size//2
	c_rects[:,2] = box_size
	c_rects[:,3] = box_size
	h_rects[:,0] = h_coords[:,1] - h_size//2
	h_rects[:,1] = h_coords[:,0] - h_size//2
	h_rects[:,2] = h_size
	h_rects[:,3] = h_size
	return c_rects, h_rects

def fill_data_tensor(c_im, h_im, c_rects, h_rects):
	"""Initializes two input tensors (RBG & DEM) using the boxes provided by c_rects and h_rects."""
	num_candidates = c_rects.shape[0]
	input_RGB = np.zeros((num_candidates, c_box[0], c_box[1], 3), dtype=np.uint8)		# initialize tensors
	input_DEM = np.zeros((num_candidates, h_box[0], h_box[1], 1), dtype=np.uint8)
	for i in range(num_candidates):
		x_c, y_c, w_c, h_c = c_rects[i,:]
		x_h, y_h, w_h, h_h = h_rects[i,:]
		c_crop = np.copy(c_im[y_c : y_c+h_c, x_c : x_c+w_c, :]).astype(np.uint8)
		h_crop = np.copy(h_im[y_h : y_h+h_h, x_h : x_h+w_h]).astype(np.uint8)
		try:
			c_crop = cv2.resize(c_crop, (c_box[0], c_box[1]))
			h_crop = cv2.resize(h_crop, (h_box[0], h_box[1]))
			input_RGB[i,:,:,:] = c_crop
			input_DEM[i,:,:,0] = h_crop
		except:
			raise IndexError('cropping failed with c_rect = ({}, {}, {}, {}), h_rect=({}, {}, {}, {})'.format(x_c, y_c, w_c, h_c, x_h, y_h, w_h, h_h))
	return input_RGB, input_DEM

def run_on_block(c_im, h_im, padding=0, get_background=False):
	"""Run complete model on the block c_im and its corresponding height block h_im."""
	if c_im.mean() <= 1e-6:		# black part
		raise IndexError

	c_coords = proc.green_hotspots(c_im, sigma=sigma, padding=padding)							# run region proposer
	c_rects, h_rects = create_boxes(c_coords, box_size)
	input_RGB, input_DEM = fill_data_tensor(c_im, h_im, c_rects, h_rects)

	# --------
	input_RGB = proc.apply_preprocessing(input_RGB, function=proc.cielab)
	# --------

	predictions, masks = network.predict([input_RGB, input_DEM], verbose=1)							# run classification model
	masks = masks[...,0]
	crop_idxs = proc.get_class_idxs(predictions, 1)
	boxes, [confidence, masks] = c_rects[crop_idxs], [predictions[crop_idxs], masks[crop_idxs]]
	boxes, [confidence, masks] = proc.non_max_suppression(boxes, other=[confidence, masks], t=overlap_threshold)
	masks = proc.get_hard_masks(masks)

	if filter_empty_masks:
		masks, boxes, [confidence] = proc.discard_empty(masks, boxes, other=[confidence], t=crop_size_threshold)

	contours  = proc.find_contours(boxes, masks)
	centroids = proc.find_centroids(boxes, masks)

	if get_background:
		background_boxes, background_confidence = proc.get_class(c_rects, predictions, 0)
		return [contours, centroids, boxes, confidence], [background_boxes, background_confidence]
	else:
		return [contours, centroids, boxes, confidence], [[],[]]

def run_model(block_size, block_overlap=box_size, max_count=np.infty, get_background=False):
	"""Perform model on img_path by dividing it into blocks."""
	valid_blocks = get_valid_blocks(block_size, block_overlap=block_overlap, max_count=max_count)
	print(valid_blocks.keys())
	# valid_blocks = {(1,4):valid_blocks[(1,4)], (1,5):valid_blocks[(1,5)]}#, (10,23):valid_blocks[(10,23)]}

	crop_dict = dict()
	bg_dict = dict()

	for (i,j) in valid_blocks:
		block = valid_blocks[(i,j)]
		c_im, h_im = get_block(*block)

		try:
			crop_output, bg_output = run_on_block(c_im, h_im, padding=box_size)
		except Exception as e:
			print('Discarded all crops somewhere in pipeline while processing block ({},{})'.format(i,j))
			print('Exception raised: "{}"\n'.format(e))
			continue

		contours, centroids, boxes, confidence  = crop_output
		background_boxes, background_confidence = bg_output

		crop_dict[(i,j)] = {'contours'   : contours,
							'centroids'  : centroids,
							'boxes'		 : boxes,
							'confidence' : confidence,
							'block'		 : block}

		bg_dict[(i,j)] = {'background_boxes'	  : background_boxes,
						  'background_confidence' :	background_confidence,
						  'block'				  : block}

		print('Added {} crops to block ({},{})\n'.format(len(contours), i, j))

	return crop_dict, bg_dict

def remove_duplicates(center_centroids, center_contours, other_centroids, shift):
	"""Uses KDTree to remove duplicates in overlapping regions between two adjacent blocks. Shift should be a 
	tuple of the form (w, h), such that w is added to the first component of each point in other_centroids, and h
	is added to the second component. This is necessary because all points are relative to their corresponding block."""
	picks = []
	if len(other_centroids)>0:
		picks = []
		center_tree = KDTree(center_centroids)
		other_tree  = KDTree(other_centroids-np.ones(other_centroids.shape)*shift)
		q = center_tree.query_ball_tree(other_tree, overlap_distance)
		for (k, neighbour_list) in enumerate(q):
			if len(neighbour_list) < 1:
				picks.append(k)
		return center_centroids[picks], list(np.array(center_contours)[picks])
	else:
		return center_centroids, center_contours

def process_overlap(crop_dict, block_overlap):
	"""For each block, remove overlapping contour with members of its N, E and NE neighbour blocks."""
	for (i,j) in crop_dict.keys():
		contours  = crop_dict[(i,j)]['contours']
		centroids = crop_dict[(i,j)]['centroids']
		(i_ad, j_ad, height, width) = crop_dict[(i,j)]['block']

		if j>0 and (i,j-1) in crop_dict and len(centroids)>0:											# check against east block for duplicates
			centroids_e = crop_dict[(i,j-1)]['centroids']
			width_e     = crop_dict[(i,j-1)]['block'][3]
			shift_e = (width_e-2*block_overlap, 0)
			centroids, contours = remove_duplicates(centroids, contours, centroids_e, shift_e)

		if i>0 and (i-1,j) in crop_dict and len(centroids)>0:											# check against north block for duplicates
			centroids_n = crop_dict[(i-1,j)]['centroids']
			height_n    = crop_dict[(i-1,j)]['block'][2]
			shift_n = (0, height_n-2*block_overlap)
			centroids, contours = remove_duplicates(centroids, contours, centroids_n, shift_n)

		if i>0 and j>0 and (i-1,j-1) in crop_dict and len(centroids)>0:									# check against north-east block for duplicates
			centroids_ne = crop_dict[(i-1,j-1)]['centroids']
			height_ne    = crop_dict[(i-1,j-1)]['block'][2]
			width_ne     = crop_dict[(i-1,j-1)]['block'][3]
			shift_ne = (width_ne-2*block_overlap, height_ne-2*block_overlap)
			centroids, contours = remove_duplicates(centroids, contours, centroids_ne, shift_ne)

		crop_dict[(i,j)]['contours']  = contours 						# update contours & centroids
		crop_dict[(i,j)]['centroids'] = centroids
		print('Removed duplicates from block ({},{})'.format(i,j))
	print()
	return crop_dict

#============================================ Output writer ===============================================
def write_shapefiles(out_dir, block_size=500, block_overlap=box_size, max_count=np.infty, filter_edges=True, get_background=False):
	"""Writes 3 shapefiles: CONTOURS.shp, BLOCK_LINES.shp, POINTS.shp, which respectively contain crop
	contours, block shapes and crop centroids. Also writes a pickle file containing the output in dictionary form.
	This dictionary also contains the dictionary with all parameters used in the simulation under the key 'metadata'.
	The input tif is divided into overlapping blocks of size block_size+2*block_overlap.
	Duplicates in the overlap region are removed using KDTrees. The parameter max_count is included for debug purposes;
	the process is terminated after max_count blocks."""

	field_shape = fiona.open(clp_path)
	field_polygons = []
	for feature in field_shape:
		poly = shape(feature['geometry'])
		field_polygons.append(poly)
	field = MultiPolygon(field_polygons)

	crop_dict, bg_dict = run_model(block_size, block_overlap, max_count=max_count, get_background=get_background)
	crop_dict = process_overlap(crop_dict, block_overlap)

	schema_lines = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }
	schema_pnt   = { 'geometry': 'Point',   'properties': { 'name': 'str' , 'confidence':'float'} }
	schema_cnt   = { 'geometry': 'Polygon', 'properties': { 'name': 'str' , 'confidence':'float'} }

	with fiona.collection(out_dir+'CONTOURS.shp', "w", "ESRI Shapefile", schema_cnt, crs=from_epsg(4326)) as output_cnt:					# add projection
		with fiona.collection(out_dir+'POINTS.shp', "w", "ESRI Shapefile", schema_pnt, crs=from_epsg(4326)) as output_pnt:
			with fiona.collection(out_dir+'BLOCK_LINES.shp', "w", "ESRI Shapefile", schema_lines, crs=from_epsg(4326)) as output_lines:

				for (i,j) in crop_dict:
					contours  = crop_dict[(i,j)]['contours']
					centroids = crop_dict[(i,j)]['centroids']
					probs 	  = crop_dict[(i,j)]['confidence']
					(i_ad, j_ad, height, width) = crop_dict[(i,j)]['block']

					count = 0
					for (k, cnt) in enumerate(contours):							# write contours
						xs, ys = cnt[:,1] + j_ad, cnt[:,0] + i_ad
						centroid = (centroids[k,0] + j_ad, centroids[k,1] + i_ad)
						transformed_contour  = Polygon([transform*(xs[l], ys[l]) for l in range(len(xs))])
						transformed_centroid = Point(transform*centroid)
						try:
							if transformed_contour.difference(field).is_empty or not filter_edges:			# if contour is complete enclosed in field
								output_cnt.write({'properties': { 'name': '({},{}): {}'.format(i, j, k), 'confidence':float(max(probs[k]))},
							            		  'geometry': mapping(transformed_contour)})
								output_pnt.write({'properties': { 'name': '({},{}): {}'.format(i, j, k), 'confidence':float(max(probs[k]))},
								            	  'geometry': mapping(transformed_centroid)})
								count += 1
							else:
								print('Crop ({},{}):{} intersects field edge'.format(i,j,k))
						except:
							print('Contour ({},{}):{} invalid'.format(i,j,k))
					print('{} crops written to block ({},{})'.format(count,i,j))

					block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
					transformed_vertices = [transform*(a,b) for (b,a) in block_vertices]
					output_lines.write({'properties' : {'name': 'block ({},{})'.format(i,j)},
										'geometry' : mapping(Polygon(transformed_vertices))})

	params['input_tif'] = img_path
	params['input_dem'] = dem_path
	params['input_clp'] = clp_path
	crop_dict['metadata'] = params

	with open(out_dir+'DATA.pickle', 'wb') as file:
		pickle.dump(crop_dict, file)

	if get_background:
		with open(out_dir+'BG_DATA.pickle', 'wb') as bg_file:
			pickle.dump(bg_dict, bg_file)

	print('\nFinished!')

if __name__ == "__main__":
	if platform == 'linux':
		img_name = img_path.split(r'/')[-1].split('.')[0]								# name
		out_directory = '/'.join(img_path.split('/')[:-1])+'/Plant Count/'				# place folder Plant Count in the same folder as img_path
	elif platform == 'windows':
		img_name = img_path.split(r"\\")[-1].split('.')[0]
		out_directory = r"../PLANT COUNT - "+img_name+r"\\"
	if not os.path.exists(out_directory):
	    os.makedirs(out_directory)
	write_shapefiles(out_directory, block_size=block_size, block_overlap=block_overlap, get_background=True, max_count=10)#, max_count=10)#, get_background=True)
