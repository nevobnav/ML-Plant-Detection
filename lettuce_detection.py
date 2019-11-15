#!/usr/bin/python3.6
platform = 'linux'

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
params = settings.get_settings('lettuce', box_size=40)
for param in params.keys():												# load all non-string parameters
	if type(params[param]) != str:
		exec('{}={}'.format(param, params[param]))

#========================================== Get Parameters & Models ==========================================
if platform == 'linux':
	name = 'c08_biobrass-C49-201906151558'
	GR = True
	img_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"
elif platform == 'windows':
	img_path = r"D:\\Old GR\\c01_verdonk-Rijweg stalling 1-201907230859-GR.tif"
	dem_path = r"D:\\Old GR\\c01_verdonk-Rijweg stalling 1-201907230859_DEM-GR.tif"
	clp_path = r"Field Shapefiles\\c01_verdonk-Rijweg stalling 1-201907230859-GR_FIELD.shp"

dem_functions 	 = tif_functions.get_functions_rasterio(img_path, dem_path, clp_path)		# functions to jump between color image and heightmap
get_adj_window	 = dem_functions['get_adjusted_window']
get_block 		 = dem_functions['get_block']
dem_scale_factor = dem_functions['scale_factor']						# constant

def windows_load(path):
	name = path.split(r'\\')[-1].split('.')[0]
	with open(path, 'r') as f:
		json_string = f.read()
	new_model = ker_models.model_from_json(json_string)
	new_model.load_weights(path.split('.')[0]+'_weights.h5')
	return new_model

if platform == 'linux':
	box_model  = ker_models.load_model(params['detection_model_path'])
	mask_model = ker_models.load_model(params['masking_model_path'])
	mask_model_dark = ker_models.load_model(params['masking_model_path_dark'])
	mask_models = (mask_model_dark, mask_model)
elif platform == 'windows':
	from keras.utils import CustomObjectScope
	from keras.initializers import glorot_uniform
	with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		box_model = windows_load(params['detection_model_path'].replace('h5','json').replace('/', r'\\'))
		mask_model = windows_load(params['masking_model_path'].replace('h5','json').replace('/', r'\\'))

#============================================ Model functions ===============================================
def create_boxes(c_coords):
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
	num_candidates = c_rects.shape[0]
	c_crops = np.zeros((num_candidates, c_box[0], c_box[1], 3), dtype=np.uint8)		# initialize tensors
	h_crops = np.zeros((num_candidates, h_box[0], h_box[1], 1), dtype=np.uint8)
	for i in range(num_candidates):
		x_c, y_c, w_c, h_c = c_rects[i,:]
		x_h, y_h, w_h, h_h = h_rects[i,:]
		c_crop = np.copy(c_im[y_c : y_c+h_c, x_c : x_c+w_c, :]).astype(np.uint8)
		h_crop = np.copy(h_im[y_h : y_h+h_h, x_h : x_h+w_h]).astype(np.uint8)
		try:
			c_crop = cv2.resize(c_crop, (c_box[0], c_box[1]))
			h_crop = cv2.resize(h_crop, (h_box[0], h_box[1]))
			c_crops[i,:,:,:] = c_crop
			h_crops[i,:,:,0] = h_crop
		except:
			raise IndexError('cropping failed with c_rect = ({}, {}, {}, {}), h_rect=({}, {}, {}, {})'.format(x_c, y_c, w_c, h_c, x_h, y_h, w_h, h_h))
	return c_crops, h_crops

def run_on_block(c_im, h_im, padding=0):
	"""Run complete model on the block c_im and its corresponding height block h_im."""
	if c_im.mean() <= 2:		# black part
		return [], np.array([])
	green_centers = proc.green_hotspots(c_im, sigma=sigma, padding=padding)		# run green region proposer
	dark_centers  = proc.dark_hotspots(c_im, sigma=sigma, padding=padding)		# run dark region proposer
	c_coords = np.concatenate((green_centers, dark_centers))

	c_rects, h_rects = create_boxes(c_coords)
	c_crops, h_crops = fill_data_tensor(c_im, h_im, c_rects, h_rects)

	predictions = box_model.predict([c_crops, h_crops], verbose=1)								# run classification model
	sorted_predictions = proc.multi_class_sort(c_rects, predictions)
	output = [([],[],[]) for k in range(len(sorted_predictions))]

	for (k, (rects, probs)) in enumerate(sorted_predictions):
		if len(rects) > 0:
			rects, probs = proc.non_max_suppression(rects, probs=probs, t=overlap_threshold)
			masks = proc.get_masks(rects, c_im, mask_models[k], verbose=1)
			output[k] = (rects, probs, masks)

	new_output = [([],[]) for k in range(len(output))]
	for (k, (rects, probs, masks)) in enumerate(output):
		try:
			if filter_empty_masks:
				rects, probs, masks = proc.discard_empty(rects, probs, masks, t=crop_size_threshold)
			if filter_disjoint:
				masks = proc.remove_unconnected_components(masks)
			if recenter:
				rects, altered = proc.recenter_boxes(rects, masks, d=center_distance)		# indeces of moved boxes
				new_masks = proc.get_masks(rects[altered], c_im, mask_models[k], verbose=1)			# compute new masks of moved boxes
				if filter_disjoint:
					new_masks = proc.remove_unconnected_components(new_masks)
				masks[altered] = new_masks																# set new masks
				if filter_empty_masks:
					rects, probs, masks = proc.discard_empty(rects, probs, masks, t=crop_size_threshold)

			cnts  = proc.find_contours(rects, masks)
			ctrs = proc.find_centroids(rects, masks)
			new_output[k] = (cnts, ctrs)
		except:
			print('No {} lettuce found'.format(['drk', 'grn'][k]))

	return new_output

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
	trans    = tif.transform
	tif.close()

	valid_blocks = dict()
	count = 0
	for i in range(0,num_rows+1):
		for j in range(0,num_cols+1):
			i_ad, j_ad, height, width = get_adj_window(i*block_size-block_overlap, j*block_size-block_overlap,
													   block_size+2*block_overlap, block_size+2*block_overlap)
			block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
			transformed_vertices = Polygon([trans*(a,b) for (b,a) in block_vertices])
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

def run_model(block_size, block_overlap=box_size, max_count=np.infty):
	"""Perform model on img_path by dividing it into blocks."""
	valid_blocks = get_valid_blocks(block_size, block_overlap=block_overlap, max_count=max_count)
	# valid_blocks = {(0,0):valid_blocks[(2,30)], (0,1):valid_blocks[(2,31)], (0,2):valid_blocks[(2,32)]}
	data_dict = dict()

	for (i,j) in valid_blocks:
		i_ad, j_ad, height, width = valid_blocks[(i,j)]
		c_im, h_im = get_block(i_ad, j_ad, height, width)
		if height<=2*block_overlap or width<=2*block_overlap:					# block too small to incorporate overlap
			continue

		try:
			print('Block size: {} x {}'.format(c_im.shape[0], c_im.shape[1]))
			(drk_cnts, drk_ctrs), (grn_cnts, grn_ctrs) = run_on_block(c_im, h_im, padding=box_size)
		except:
			print('No crops found in block ({},{})'.format(i,j))
			continue

		data_dict[(i,j)] = {'contours':  (drk_cnts, grn_cnts), 
							'centroids': (drk_ctrs, grn_ctrs), 
							'block': (i_ad, j_ad, height, width)}
		# print('Added {} crops to block ({},{})\n'.format(len(contours), i, j))

	return data_dict

def process_overlap(data_dict, block_overlap):
	"""Uses KDTree to remove duplicates in overlapping regions between two adjacent blocks."""
	for (i,j) in data_dict.keys():
		reg_contours, drk_contours  = data_dict[(i,j)]['contours']
		reg_centroids, drk_centroids = data_dict[(i,j)]['centroids']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']

		if j>0 and (i,j-1) in data_dict:											# check against east block for duplicates
			reg_centroids_e, drk_centroids_e = data_dict[(i,j-1)]['centroids']
			width_e = data_dict[(i,j-1)]['block'][3]
			shift_e = (width_e-2*block_overlap, 0)
			if len(reg_centroids) > 0:
				reg_centroids, reg_contours = proc.remove_duplicates(reg_centroids, reg_contours, reg_centroids_e, shift_e, overlap_distance=overlap_distance)
			if len(drk_centroids) > 0:
				drk_centroids, drk_contours = proc.remove_duplicates(drk_centroids, drk_contours, drk_centroids_e, shift_e, overlap_distance=overlap_distance)

		if i>0 and (i-1,j) in data_dict:											# check against north block for duplicates
			reg_centroids_n, drk_centroids_n = data_dict[(i-1,j)]['centroids']
			height_n = data_dict[(i-1,j)]['block'][2]
			shift_n  = (0, height_n-2*block_overlap)
			if len(reg_centroids) > 0:
				reg_centroids, reg_contours = proc.remove_duplicates(reg_centroids, reg_contours, reg_centroids_n, shift_n, overlap_distance=overlap_distance)
			if len(drk_centroids) > 0:
				drk_centroids, drk_contours = proc.remove_duplicates(drk_centroids, drk_contours, drk_centroids_n, shift_n, overlap_distance=overlap_distance)

		if i>0 and j>0 and (i-1,j-1) in data_dict:									# check against north-east block for duplicates
			reg_centroids_ne, drk_centroids_ne = data_dict[(i-1,j-1)]['centroids']
			height_ne = data_dict[(i-1,j-1)]['block'][2]
			width_ne  = data_dict[(i-1,j-1)]['block'][3]
			shift_ne = (width_ne-2*block_overlap, height_ne-2*block_overlap)
			if len(reg_centroids) > 0:
				reg_centroids, reg_contours = proc.remove_duplicates(reg_centroids, reg_contours, reg_centroids_ne, shift_ne, overlap_distance=overlap_distance)
			if len(drk_centroids) > 0:
				drk_centroids, drk_contours = proc.remove_duplicates(drk_centroids, drk_contours, drk_centroids_ne, shift_ne, overlap_distance=overlap_distance)

		data_dict[(i,j)]['contours']  = (reg_contours, drk_contours) 						# update contours & centroids
		data_dict[(i,j)]['centroids'] = (reg_centroids, drk_centroids)
		print('Removed duplicates from block ({},{})'.format(i,j))
	print()
	return data_dict

#============================================ Output writer ===============================================
def write_shapefiles(out_dir, block_size=500, block_overlap=box_size, max_count=np.infty):
	"""Writes 3 shapefiles: CONTOURS.shp, BLOCK_LINES.shp, POINTS.shp, which respectively contain crop
	contours, block shapes and crop centroids. The tif is divided into overlapping blocks of size block_size+2*block_overlap.
	Duplicates in the overlap region are removed using KDTrees. The parameter max_count is included for debug purposes;
	the process is terminated after max_count blocks."""
	tif = rasterio.open(img_path)
	trans = tif.transform
	tif.close()

	filter_edges = True

	field_shape = fiona.open(clp_path)
	field_polygons = []
	for feature in field_shape:
		poly = shape(feature['geometry'])
		field_polygons.append(poly)
	field = MultiPolygon(field_polygons)

	data_dict = run_model(block_size, block_overlap, max_count=max_count)
	data_dict = process_overlap(data_dict, block_overlap)
	with open(out_dir+'DATA.pickle', 'wb') as file:
		pickle.dump(data_dict, file)

	schema_lines = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }
	schema_pnt   = { 'geometry': 'Point',   'properties': { 'name': 'str' } }
	schema_cnt   = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }

	with fiona.collection(out_dir+'CONTOURS.shp', "w", "ESRI Shapefile", schema_cnt, crs=from_epsg(4326)) as output_cnt:					# add projection
		with fiona.collection(out_dir+'POINTS.shp', "w", "ESRI Shapefile", schema_pnt, crs=from_epsg(4326)) as output_pnt:
			with fiona.collection(out_dir+'BLOCK_LINES.shp', "w", "ESRI Shapefile", schema_lines, crs=from_epsg(4326)) as output_lines:

				for (i,j) in data_dict:
					(drk_cnts, grn_cnts) = data_dict[(i,j)]['contours']
					(drk_ctrs, grn_ctrs) = data_dict[(i,j)]['centroids']
					(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']

					count = 0
					for name, contours, centroids in zip(('drk', 'grn'), (drk_cnts, grn_cnts), (drk_ctrs, grn_ctrs)):
						for (k, cnt) in enumerate(contours):							# write contours
							xs, ys = cnt[:,1] + j_ad, cnt[:,0] + i_ad
							centroid = (centroids[k,0] + j_ad, centroids[k,1] + i_ad)
							transformed_points   = Polygon([trans*(xs[l],ys[l]) for l in range(len(xs))])
							transformed_centroid = Point(trans*centroid)
							try:
								if transformed_points.difference(field).is_empty or not filter_edges:			# if contour is complete enclosed in field
									output_pnt.write({'properties': { 'name': '({},{}): {}: {}'.format(i,j, k, name)},
							            			  'geometry': mapping(transformed_centroid)})
									output_cnt.write({'properties': { 'name': '({},{}): {}'.format(i, j, name)},
								            			  'geometry': mapping(transformed_points)})
									count += 1
								else:
									print('Crop ({},{}):{} intersects field edge'.format(i,j,k))
							except:
								print('Contour ({},{}):{} invalid'.format(i,j,k))
						print('{} crops written to block ({},{})'.format(count,i,j))

					block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
					transformed_vertices = [trans*(a,b) for (b,a) in block_vertices]
					output_lines.write({'properties' : {'name': 'block ({},{})'.format(i,j)},
										'geometry' : mapping(Polygon(transformed_vertices))})

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
	write_shapefiles(out_directory, block_size=block_size, block_overlap=block_overlap, max_count=20)
