#!/usr/bin/python3.6

"""
Script containing all external processing functions, such as region
proposal methods, class sorting methods, contour extraction etc.
"""

import numpy as np
import cv2
from scipy.ndimage import filters
from scipy.ndimage import measurements
from skimage import measure
import skimage.color
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

# ========================================= Region Proposal =============================================
def green_hotspots(im, sigma=4, padding=0, m=2):
	"""RoI generator based on green hotspots.

	The input is first converted from RGB to YUV. Then the U and V channels
	are added and smoothed with a Gaussian filter. The function peak_local_max
	from scipy.spatial is used to find local minimi in the resulting array.

	Arguments
	---------
	im : (?,?,3) numpy array
		Input RGB image.
	sigma : float (>0), optional
		Gaussian filter smoothing parameter. Decreasing sigma will result in
		more hotspots. The default value is 4 is a good starting point.
	padding : int, optional
		Width of boundary layer in which no hotspots will be detected. Keep at
		roughly the box_size to prevent boxes from being clipped by block 
		boundary. Default is 0.
	m : int, optional
		Size of minimum filter. Keep at the default of 2. 

	Returns
	-------
	coords : (?,2) numpy array
		Array containing coordinates of the green hotspots.
	"""

	if padding > 0:
		im = im[padding:-padding, padding:-padding, :]
	im_yuv = skimage.color.rgb2yuv(im)
	im_filtered = im_yuv[:,:,1] + im_yuv[:,:,2]
	im_filtered = filters.minimum_filter(im_filtered, m)
	im_filtered = filters.gaussian_filter(im_filtered, sigma)
	mask = peak_local_max(-im_filtered, indices=False)
	mask = np.logical_and(mask, im_filtered!=0)
	coords = np.argwhere(mask==1)
	return coords+padding

def dark_hotspots(im, sigma=6, padding=0, m=2):
	"""RoI generator based on dark hotspots.

	Similar function to green_hotspots. Use this function to detect locations
	of dark lettuce crops. It is based on local minima of the Y band of 
	image converted to YUV.

	Arguments
	---------
	im : (?,?,3) numpy array
		Input RGB image.
	sigma : float (>0), optional
		Gaussian filter smoothing parameter. Decreasing sigma will result in
		more hotspots. The default value is 6 is a good starting point.
	padding : int, optional
		Width of boundary layer in which no hotspots will be detected. Keep at
		roughly the box_size to prevent boxes from being clipped by block 
		boundary. Default is 0.
	m : int, optional
		Size of minimum filter. Keep at the default of 2. 

	Returns
	-------
	coords : (?,2) numpy array
		Array containing coordinates of the dark hotspots.
	"""
	if padding > 0:
		im = im[padding:-padding, padding:-padding, :]
	im_yuv = skimage.color.rgb2yuv(im)
	im_filtered = im_yuv[:,:,0]
	im_filtered = filters.minimum_filter(im_filtered, m)
	im_filtered = filters.gaussian_filter(im_filtered, sigma)
	mask = peak_local_max(-im_filtered, indices=False)
	mask = np.logical_and(mask, im_filtered!=0)
	coords = np.argwhere(mask==1)
	return coords+padding

# =========================================== Box filters ===============================================
def get_class_idxs(predictions, class_index):
	"""Get indeces of boxes belonging to class.

	Arguments
	---------
	predictions : (N,num_classes) numpy array
		Array containing confidence scores of a box belonging to class.
	class_index : int
		Index of class, should be either 0, 1, ..., num_classes-1.

	Returns
	-------
	idxs : list
		List containing indeces of elements belonging to class given by
		class_index
	"""

	num_candidates = predictions.shape[0]
	idxs = []
	for i in range(num_candidates):
		pred_index = np.argmax(predictions[i,:])		# prediction index corresponds to label name
		if pred_index == class_index:
			idxs.append(i)					# store index belonging to predicted broccoli
	return idxs

def non_max_suppression(boxes, other=[], t=0.2):
	"""Non-Max-Suppresion algorithm. 

	Arguments
	---------
	boxes : (N,4) numpy array
		Array containin box locations and dimensions, each rows is
		of the form [x, y, w, h].
	other : list, optional
		List containing other objects of length N, from which elements
		must also be removed. Default is [].
	t : float (in [0,1])
		Intersection over Union (IoU) threshold. Boxes that have a bigger
		overlap than t are removed.

	Returns
	-------
	new_boxes : (N*,4) numpy array
		Array containing boxes which have not been removed, note N*<=N.
	if other!=[], also returns a list of filtered objects in other.
	"""

	pick = []										# list containing indeces of boxes that we want to keep
	x1 = boxes[:,0]									# x1 = x
	y1 = boxes[:,1]									# y1 = y
	x2 = boxes[:,0]+boxes[:,2]						# x2 = x + w
	y2 = boxes[:,1]+boxes[:,3]						# y2 = y + h
	area = (x2 - x1 + 1) * (y2 - y1 + 1)			# bias area to avoid division by zero.
	idxs = np.argsort(y2)							# return array of indeces such that y2[idxs] is sorted
	while idxs.shape[0] > 0:
		last = idxs.shape[0] - 1
		i = idxs[last]
		pick.append(i)
		xx1 = np.maximum(x1[i], x1[idxs[:last]])	# clever way of computing overlaps
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > t)[0])))		# delete last b/c we checked it, and delete all indeces for which the overlap is large
	if len(other) > 0:
		return boxes[pick], [o[pick] for o in other]					# list as index only works with numpy arrays
	else:
		return boxes[pick]

# ======================================= Masking Functions ===========================================
def discard_empty(masks, rects, other=[], t=0.01):
	"""Discards boxes which are nearly empty.

	Arguments
	---------
	masks : (N,?,?) numpy array
		Tensor containing masks.
	rects : (N, 4) numpy array
		Array containing box locations and dimensions.
	other : list
		List containing other arrays of length N that should be filtered.
	t : float (in [0,1])
		Area ratio threshold. If area of mask is smaller thant*box_area,
		discard it.
	"""

	crop_areas = np.sum(np.sum(masks, axis=2), axis=1)#[:,0]
	is_almost_empty = crop_areas < t*rects[:,2]*rects[:,3]
	filtered_masks = np.array([elt for (i,elt) in enumerate(masks) if not is_almost_empty[i]])
	filtered_rects = np.array([elt for (i,elt) in enumerate(rects) if not is_almost_empty[i]])
	for (idx, o) in enumerate(other):
		other[idx] = np.array([elt for (i,elt) in enumerate(o) if not is_almost_empty[i]])
	if len(other) > 0:
		return filtered_masks, filtered_rects, other
	else:
		return filtered_masks, filtered_rects

def get_hard_masks(masks, sigma=2):
	"""Converts soft masks to hard, smoothed, singularly connected masks.

	First smooths masks with Gaussian filter, then converts to binary (hard)
	mask. Small unconnected components are removed.

	Arguments
	---------
	masks : (N,?,?) numpy array
		Tensor containing smooth masks.
	sigma : float, optional
		Smoothing parameter of Gaussian filter.

	Returns
	-------
	hard_masks : (N,?,?) numpy array
		Tensor containing generated hard masks.
	"""

	new_masks = np.zeros(masks.shape)
	for (i, mask) in enumerate(masks):
		mask = filters.gaussian_filter(mask, sigma)>0.5
		labelled_mask, num_components = measurements.label(mask)
		areas = []
		if num_components > 1:
			for k in range(1, num_components+1):
				area = (labelled_mask==k).sum()
				areas.append(area)
			n = np.argmax(areas)+1
			new_masks[i,:,:] = labelled_mask==n
		else:
			new_masks[i,:,:] = mask
	return new_masks

def find_contours(rects, masks):
	"""Finds contour around mask in each box in rects. 

	If no contour can be generated due to the mask being of 
	poor quality, a rectangular contour will be added.

	Arguments
	---------
	rects : (N,4) numpy array
		Array containing rectangles.
	masks : (N,?,?) numpy array
		Tensor containing masks. Each slice [i,:,:] is one mask.

	Returns
	-------
	contours : list
		List containing (K,2) numpy arrays, where the columns of 
		each array represent the x- and y-coordinates of a contour.
	"""

	contours = []
	for (i, rect) in enumerate(rects):
		x, y, w, h = rect
		mask = masks[i,...] # cv2.resize(masks[i,...], (w, h))
		try:
			rel_cnt = measure.find_contours(mask, 0.5)[0].astype(float)
			rel_cnt[:,0] = rel_cnt[:,0]*w/mask.shape[1]
			rel_cnt[:,1] = rel_cnt[:,1]*h/mask.shape[0]
			rel_cnt[:,0] += y
			rel_cnt[:,1] += x
			contours.append(rel_cnt)
		except IndexError:
			contours.append(np.array([0,0],[0,w],[h,w],[h,0]))
	return contours

def find_centroids(rects, masks):
	"""Finds the centroid of each mask. 

	If the mask is empty, the centroid will be placed in the 
	middle of the box.

	Arguments
	---------
	rects : (N,4) numpy array
		Array containing rectangles.
	masks : (N,?,?) numpy array
		Tensor containing masks. Each slice [i,:,:] is one mask.

	Returns
	-------
	centroids : (K,2) numpy arry
		Each row represents one point.
	"""

	box_size = rects[0,2]
	mask_width, mask_height = masks.shape[1], masks.shape[2]
	centroids = np.zeros((masks.shape[0], 2))
	xs = np.arange(masks.shape[1])
	ys = np.arange(masks.shape[2])
	for (i, mask) in enumerate(masks):
		# mask = cv2.resize(mask.astype(np.uint8), box_size)
		x, y, w, h = rects[i,:]
		area = mask.sum()
		if area==0:
			centroids[i,:] = [x+w//2, y+h//2]
		else:
			xc = np.sum(xs*mask)*box_size/mask_width
			yc = np.sum(ys*mask.T)*box_size/mask_height
			centroids[i,:] = [x+xc/area, y+yc/area]
	return centroids

# ==================================== Overlapping Crops ========================================
def remove_duplicates(center_centroids, center_contours, other_centroids, shift, min_distance):
	"""Removes duplicate contours and centroids from center_contours and 
	center_centroids.

	Uses KDTree to remove duplicates in overlapping regions between two 
	adjacent blocks. The objects in the other block are shifted, This is 
	necessary because all points are relative to their corresponding block.

	Arguments
	---------
	center_centroids : (N,2) numpy array
		Array containing points relative to the center block.
	center_contours : list of length N
		List with contours corresponding to points in center block.
	other_centroids : (M,2) numpy array
		Array containing points relative to other block.
	shift : 2-tuple
		Tuple of the form (w, h), such that w is added to the first component 
		of each point in other_centroids, and h is added to the second component.
	min_distance : float
		Minimum distance between to centroids. If their distance is smaller,
		remove the centroid and its corresponding contour from the center block.

	Returns
	-------
	center_centroids : (N*,2) numpy array
		Numpy array containing points which have not been removed, note that
		N* <= N.
	center_contours : list of length N*
		List containing contours corresponding to the points which have not
		been removed.
	"""

	picks = []
	if len(other_centroids)>0:
		picks = []
		center_tree = KDTree(center_centroids)
		other_tree  = KDTree(other_centroids-np.ones(other_centroids.shape)*shift)
		q = center_tree.query_ball_tree(other_tree, min_distance)
		for (k, neighbour_list) in enumerate(q):
			if len(neighbour_list) < 1:
				picks.append(k)
		return center_centroids[picks], list(np.array(center_contours)[picks])
	else:
		return center_centroids, center_contours

def process_overlap(crop_dict, num_classes, block_overlap, min_distance):
	"""Removes duplicates in each block.

	Iterates over all blocks in crop_dict, and removes duplicates with its
	north, east and north-east neighbour blocks using the function remove_duplicates.

	Arguments
	---------
	crop_dict : dict
		Dictionary containing data for each block. It should have the same
		structure as the attribute detected_crops of a Detector class.
	block_overlap : int
		Width of overlap region between two blocks.
	min_distance : float
		Minimum distance between to centroids. If their distance is smaller,
		the centroid and its corresponding contour are removed.

	Returns
	-------
	crop_dict : dict
		Updated dictionary.
	"""

	for (i,j) in crop_dict.keys():
		(i_ad, j_ad, height, width) = crop_dict[(i,j)]['block']

		if j>0 and (i,j-1) in crop_dict:
			width_e = crop_dict[(i,j-1)]['block'][3]
			shift_e = (width_e-block_overlap, 0)
			for class_idx in range(1, num_classes):
				contours  = crop_dict[(i,j)][class_idx]['contours']
				centroids = crop_dict[(i,j)][class_idx]['centroids']	
				if len(centroids)>0:								# check against east block for duplicates
					centroids_e = crop_dict[(i,j-1)][class_idx]['centroids']
					centroids, contours = remove_duplicates(centroids, contours, centroids_e, shift_e, min_distance)
					crop_dict[(i,j)][class_idx]['contours'] = contours
					crop_dict[(i,j)][class_idx]['centroids'] = centroids

		if i>0 and (i-1,j) in crop_dict:											# check against north block for duplicates
			height_n = crop_dict[(i-1,j)]['block'][2]
			shift_n = (0, height_n-block_overlap)
			for class_idx in range(1, num_classes):
				contours  = crop_dict[(i,j)][class_idx]['contours']
				centroids = crop_dict[(i,j)][class_idx]['centroids']
				if len(centroids)>0:											# check against east block for duplicates
					centroids_n = crop_dict[(i-1,j)][class_idx]['centroids']
					centroids, contours = remove_duplicates(centroids, contours, centroids_n, shift_n, min_distance)
					crop_dict[(i,j)][class_idx]['contours'] = contours
					crop_dict[(i,j)][class_idx]['centroids'] = centroids

		if i>0 and j>0 and (i-1,j-1) in crop_dict:									# check against north-east block for duplicates
			height_ne = crop_dict[(i-1,j-1)]['block'][2]
			width_ne  = crop_dict[(i-1,j-1)]['block'][3]
			shift_ne = (width_ne-block_overlap, height_ne-block_overlap)
			for class_idx in range(1, num_classes):
				contours  = crop_dict[(i,j)][class_idx]['contours']
				centroids = crop_dict[(i,j)][class_idx]['centroids']
				if len(centroids)>0:											# check against east block for duplicates
					centroids_ne = crop_dict[(i-1,j-1)][class_idx]['centroids']
					centroids, contours = remove_duplicates(centroids, contours, centroids_ne, shift_ne, min_distance)
					crop_dict[(i,j)][class_idx]['contours'] = contours
					crop_dict[(i,j)][class_idx]['centroids'] = centroids

		print('Removed duplicates from block ({},{})'.format(i,j))

	return crop_dict