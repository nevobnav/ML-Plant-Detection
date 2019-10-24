#!/usr/bin/python3.6
#================================================== Imports ==================================================
import os
import cv2
import tensorflow.keras.models as ker_models
import rasterio
from shapely.geometry import Polygon, Point, mapping
from scipy.spatial import KDTree
from fiona import collection
import numpy as np 

import processing as proc
import dem_functions
import settings

#================================================= Crop Type =================================================
crop = 'lettuce'														# which crop is present in field
params = settings.get_settings(crop, block_size=1000)

#============================================== Get Parameters ===============================================
for param in params.keys():												# load all non-string parameters
	if type(params[param]) != str:
		exec('{}={}'.format(param, params[param]))

# img_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c01_verdonk-Rijweg stalling 2-201907170908-GR_cropped.tif'
# dem_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c01_verdonk-Rijweg stalling 2-201907170908_DEM-GR.tif'

# img_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c01_verdonk-Wever oost-201907240707-GR_cropped.tif'
# dem_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c01_verdonk-Wever oost-201907240707_DEM-GR.tif'

img_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c08_biobrass-C49-201906151558-GR_cropped.tif'
dem_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c08_biobrass-C49-201906151558_DEM-GR.tif'

dem_functions 	 = dem_functions.get_functions(img_path, dem_path)		# functions to jump between color image and heightmap
get_window 		 = dem_functions['get_window']							
dem_scale_factor = dem_functions['scale_factor']						# constant

box_model  = ker_models.load_model(params['detection_model_path'])
mask_model = ker_models.load_model(params['masking_model_path'])

#================================================ Functions ==================================================
def pop(x, k): 
	"""Removes the k-th element of the array x. Removal is not done in-place, to update x, use x=pop(x,k)."""
	k = k%x.shape[0] 
	return np.concatenate((x[:k], x[k+1:])) 

def run(c_im, h_im, padding=0):
	"""Run complete model on the block c_im and its corresponding height block h_im."""
	empty_output = np.zeros((c_im.shape[0], c_im.shape[1])), np.array([]), np.array([])			# if at some point there are no more boxes, return this
	c_size = box_size
	h_size = int(np.round(dem_scale_factor*c_size))

	print('Block size: {}x{}'.format(c_im.shape[0], c_im.shape[1]))
	c_coords = proc.window_hotspots_centers(c_im, sigma=sigma, padding=padding, top_left=0)		# run region proposer				
	h_coords = np.round(dem_scale_factor*c_coords).astype(int) 									# convert to DEM row/col

	c_rects = np.zeros((c_coords.shape[0], 4), dtype=int)
	h_rects = np.zeros((h_coords.shape[0], 4), dtype=int)

	c_rects[:,0] = c_coords[:,1] - c_size//2
	c_rects[:,1] = c_coords[:,0] - c_size//2
	c_rects[:,2] = c_size
	c_rects[:,3] = c_size

	h_rects[:,0] = h_coords[:,1] - h_size//2
	h_rects[:,1] = h_coords[:,0] - h_size//2
	h_rects[:,2] = h_size
	h_rects[:,3] = h_size

	num_candidates = c_rects.shape[0]														# number of rectangles to be analyzed
	if num_candidates==0:
		return empty_output

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

	predictions = box_model.predict([c_crops, h_crops], verbose=1)							# run classification model
	(back_rects, back_probs), (pre_broc_rects, pre_broc_probs) = proc.sort_into_classes(c_rects, predictions)
	broc_rects, broc_probs = proc.non_max_suppression(pre_broc_rects, probs=pre_broc_probs, t=overlap_threshold)

	if len(broc_rects)==0:
		return empty_output
	masks = proc.get_masks(broc_rects, c_im, mask_model, verbose=1)

	if filter_masks:
		broc_rects, broc_probs, masks = proc.discard_empty(broc_rects, broc_probs, masks, t=crop_size_threshold)
		if len(broc_rects)==0:
			return empty_output

	if filter_disjoint:
		masks = proc.remove_unconnected_components(masks)

	if recenter:
		broc_rects, altered = proc.recenter_boxes(broc_rects, masks, d=center_distance)		# indeces of moved boxes
		if len(broc_rects)==0:
			return empty_output
		new_masks = proc.get_masks(broc_rects[altered], c_im, mask_model, verbose=1)		# compute new masks of moved boxes
		if filter_disjoint:
			new_masks = proc.remove_unconnected_components(new_masks)
		masks[altered] = new_masks															# set new masks
		if filter_masks:
			broc_rects, broc_probs, masks = proc.discard_empty(broc_rects, broc_probs, masks, t=crop_size_threshold)
	if len(broc_rects)==0:
		return empty_output

	big_mask  = proc.create_big_mask(c_im, broc_rects, masks)
	contours  = proc.find_contours(broc_rects, masks)
	centroids = proc.find_centroids(broc_rects, masks)

	return big_mask, contours, centroids

def write_shapefiles(out_dir, block_size=500, block_overlap=box_size, max_count=np.infty):
	"""Writes 3 shapefiles: out_name_CONTOURS, out_name_LINES, out_name_POINTS, which respectively contain crop
	contours, block shape and crop centroids. The tif is divided into blocks of size block_size. Two adjacent blocks
	have overlap of 2*block_overlap. Duplicates in the overlap region are removed using KDTrees. The parameter 
	max_count is included for debug purposes; the process is terminated after max_count blocks."""
	tif = rasterio.open(img_path)
	profile  = tif.profile.copy()
	trans    = tif.transform
	num_cols = tif.width//block_size
	num_rows = tif.height//block_size
	tif.close()

	count = 0
	padding = box_size
	data_dict = dict()

	for i in range(0,num_rows+1):
		for j in range(0,num_cols+1):
			try:
				c_im, h_im, (i_ad, j_ad, height, width) = get_window(i*block_size-block_overlap, j*block_size-block_overlap, 
																	 block_size+2*block_overlap, block_size+2*block_overlap)
				if height<=2*block_overlap or width<=2*block_overlap:					# block too small to incorporate overlap
					continue
			except:
				continue

			try:
				mask, contours, centroids = run(c_im, h_im, padding=padding)
			except:
				mask, contours, centroids = np.zeros((c_im.shape[0], c_im.shape[1])), np.array([]), np.array([])			

			data_dict[(i,j)] = {'contours':contours, 'centroids':centroids, 'block':(i_ad, j_ad, height, width)}

			count += 1
			print('Block ({},{}) complete: {}/{}\n'.format(i, j, count, (num_cols+1)*(num_rows+1)))

			if count >= max_count:
				break
		if count >= max_count:
			break

	schema_lines = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }
	schema_pnt   = { 'geometry': 'Point',   'properties': { 'name': 'str' } }
	schema_cnt   = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }

	for (i,j) in data_dict.keys():
		contours  = data_dict[(i,j)]['contours']
		centroids = data_dict[(i,j)]['centroids']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']

		if j>0 and len(centroids)>0:									# check against east block for duplicates
			east_contours  = data_dict[(i,j-1)]['contours']
			east_centroids = data_dict[(i,j-1)]['centroids']
			width_east     = data_dict[(i,j-1)]['block'][3]

			if len(east_centroids)>0:
				center_tree = KDTree(centroids)
				east_tree   = KDTree(east_centroids-np.ones(east_centroids.shape)*(width_east-2*block_overlap,0))
				q = center_tree.query_ball_tree(east_tree, overlap_distance)
				picks = []
				for (k, neighbour_list) in enumerate(q):
					if len(neighbour_list) >= 1:
						picks.append(k)
				for k in picks[::-1]:
					centroids = pop(centroids, k)
					contours.pop(k)

		if i>0 and len(centroids)>0:									# check against north block for duplicates
			north_contours  = data_dict[(i-1,j)]['contours']
			north_centroids = data_dict[(i-1,j)]['centroids']
			height_north    = data_dict[(i-1,j)]['block'][2]

			if len(north_centroids)>0:
				center_tree = KDTree(centroids)
				north_tree  = KDTree(north_centroids-np.ones(north_centroids.shape)*(0,height_north-2*block_overlap))
				q = center_tree.query_ball_tree(north_tree, overlap_distance)
				picks = []
				for (k, neighbour_list) in enumerate(q):
					if len(neighbour_list) >= 1:
						picks.append(k)
				for k in picks[::-1]:
					centroids = pop(centroids,k)
					contours.pop(k)

		data_dict[(i,j)]['contours']  = contours 						# update contours & centroids
		data_dict[(i,j)]['centroids'] = centroids
		print('Removed duplicates from block ({},{})'.format(i,j))
	print()

	with collection(out_dir+'CONTOURS.shp', "w", "ESRI Shapefile", schema_cnt) as output_cnt:
		with collection(out_dir+'POINTS.shp', "w", "ESRI Shapefile", schema_pnt) as output_pnt:
			with collection(out_dir+'BLOCK_LINES.shp', "w", "ESRI Shapefile", schema_lines) as output_lines:

				for (i,j) in data_dict.keys():
					contours  = data_dict[(i,j)]['contours']
					centroids = data_dict[(i,j)]['centroids']
					(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']

					for (k, cnt) in enumerate(contours):							# write contours
						xs, ys = cnt[:,1] + j_ad, cnt[:,0] + i_ad
						transformed_points = [trans*(xs[l],ys[l]) for l in range(len(xs))]
						poly = Polygon(transformed_points)
						output_cnt.write({
				            'properties': { 'name': '({},{}): {}'.format(i, j, k)},
				            'geometry': mapping(Polygon(transformed_points))})

					for (k, centroid) in enumerate(centroids):						# write centroids
						centroid = (centroids[k,0] + j_ad, centroids[k,1] + i_ad)
						transformed_centroid = trans*centroid
						output_pnt.write({
							'properties': { 'name': '({},{}): {}'.format(i, j, k)},
				            'geometry': mapping(Point(transformed_centroid))})

					block_edges = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
					transformed_edges = [trans*(a,b) for (b,a) in block_edges]
					output_lines.write({											# write block edges
						'properties' : {'name': 'block ({},{})'.format(i,j)},
						'geometry' : mapping(Polygon(transformed_edges))})

					print('Block ({},{}) written'.format(i,j))
	print('\nFinished!\n')
	
if __name__ == "__main__":
	img_name = img_path.split('/')[-1].split('.')[0]
	out_directory = 'PLANT COUNT - '+img_name+'/'
	if not os.path.exists(out_directory):
	    os.makedirs(out_directory)
	write_shapefiles(out_directory, block_size=block_size, block_overlap=block_overlap, max_count=20)


