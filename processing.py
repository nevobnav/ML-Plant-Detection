#!/usr/bin/python3.6

import numpy as np
# from pre_proc import mask_filter
import cv2
import tensorflow.keras.models as models
from scipy.ndimage import filters
from scipy.ndimage import measurements
from skimage import measure
import skimage.color
from scipy.spatial import KDTree

from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay
from skimage.feature import peak_local_max

# ========================================= Region Proposal =============================================
def green_hotspots(im, sigma=4, padding=0, m=2):
	"""RoI generator based on local minima of (u+v)-channel."""
	if padding > 0:
		im = im[padding:-padding, padding:-padding, :]
	im_yuv = skimage.color.rgb2yuv(im)
	im_filtered = im_yuv[:,:,1] + im_yuv[:,:,2]
	im_filtered = filters.minimum_filter(im_filtered, m)
	im_filtered = filters.gaussian_filter(im_filtered, sigma)
	mask = peak_local_max(-im_filtered, indices=False)#, exclude_border=False)
	mask = np.logical_and(mask, im_filtered!=0)
	coords = np.argwhere(mask==1)
	return coords+padding

def dark_hotspots(im, sigma=6, padding=0, m=2):
	"""RoI generator based on local minima of (u+v)-channel."""
	if padding > 0:
		im = im[padding:-padding, padding:-padding, :]
	im_yuv = skimage.color.rgb2yuv(im)
	im_filtered = im_yuv[:,:,0] # + im_yuv[:,:,2]
	im_filtered = filters.minimum_filter(im_filtered, m)
	im_filtered = filters.gaussian_filter(im_filtered, sigma)
	mask = peak_local_max(-im_filtered, indices=False)#, exclude_border=False)
	mask = np.logical_and(mask, im_filtered!=0)
	coords = np.argwhere(mask==1)
	return coords+padding

# =========================================== Box filters ===============================================
def multi_class_sort(rects, predictions, bg_index=0):
	"""Sorts each box in rects into its class as predicted by the array predictions. Returns a tuple of the
	form ((rects_i, probs_i), ...), where probs contains the probability of the corresponding box belonging to class i."""
	num_classes = predictions.shape[1]
	num_candidates = predictions.shape[0]
	indeces = [[] for k in range(num_classes)]
	for i in range(num_candidates):
		pred_index = np.argmax(predictions[i,:])
		indeces[pred_index].append(i)
	sorted_rects = [(rects[indeces[pred_index],:], predictions[indeces[pred_index], pred_index]) \
									for pred_index in range(num_classes) if pred_index!=bg_index]
	return tuple(sorted_rects)

def get_class_idxs(predictions, class_index):
	num_candidates = predictions.shape[0]
	idxs = []
	for i in range(num_candidates):
		pred_index = np.argmax(predictions[i,:])		# prediction index corresponds to label name
		if pred_index == class_index:
			idxs.append(i)					# store index belonging to predicted broccoli
	return idxs

def get_class(rects, predictions, class_index):
	"""Returns boxes and confidence arrays of a single class, specified by class_index."""
	num_candidates = predictions.shape[0]
	idxs = []
	boxes, probs = [], []

	for i in range(num_candidates):
		pred_index = np.argmax(predictions[i,:])		# prediction index corresponds to label name
		if pred_index == class_index:
			idxs.append(i)					# store index belonging to predicted broccoli
			boxes.append(rects[i,:])
			probs.append(predictions[i, pred_index])

	return np.array(boxes), np.array(probs)

def non_max_suppression(boxes, other=[], t=0.2):
	"""Non-Max-Suppresion algorithm. boxes is an (N,4)-numpy array, where N is the number of boxes.
	One row of the array boxes should be of the form [x, y, w, h], where (x,y) is the lower left
	edge of the box, w its width and h its height."""
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
def get_masks(rects, c_im, model, verbose=1):
	"""Computes mask for each box in rects using a FCN given by pre-loaded model.
	Returns a (N, w, h) tensor, where each slice [i,:,:] is a binary image, and N is the length of rects."""
	input_tensor = np.zeros((rects.shape[0], 64, 64, 3), dtype=np.uint8)
	for (i, rect) in enumerate(rects):
		x, y, w, h = rect
		x1, x2 = max(x, 0), min(x+w, c_im.shape[1])					# pad box
		y1, y2 = max(y, 0), min(y+h, c_im.shape[0])
		crop = c_im[y1:y2, x1:x2, :].astype(np.uint8)
		input_tensor[i, :, :, :] = cv2.resize(crop, (64,64))
	predictions = model.predict(input_tensor, verbose=verbose)[:,:,:,0]>0.5
	w, h = rects[0,2], rects[0,3]
	masks = np.zeros((predictions.shape[0], w, h))
	for (i, mask) in enumerate(predictions):
		masks[i,...] = cv2.resize(predictions[i,...].astype(np.uint8), (w,h))
	return masks

def discard_empty(masks, rects, other=[], t=0.01):
	"""Discards rectangles which are nearly empty. The parameter t determines the minimum amount of the
	box that should be filled as a fraction of the total area."""
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

def recenter_boxes(rects, masks, d=0.1):
	"""If the (relative) distance from box center to mask centroid is greater than d, move box such that centroid
	is its new center."""
	mask_size = rects[0,2]				# box_size
	xs = np.linspace(0,1,mask_size)
	ys = np.linspace(0,1,mask_size)
	altered = []
	for i in range(rects.shape[0]):
		mask = masks[i,:]
		x, y, w, h = rects[i,:]
		char_len = min(w, h)
		area = mask.sum()
		xc = np.sum(xs*mask)/area 		# centroid relative to box
		yc = np.sum(ys*mask.T)/area
		if (xc-0.5)**2 + (yc-0.5)**2 >= d**2:
			rects[i,0] = x + int((xc-0.5)*w)
			rects[i,1] = y + int((yc-0.5)*h)
			altered.append(i)
	return rects, altered

def remove_unconnected_components(masks):
	"""If mask consists of multiple components, only retain the component with maximum area, and
	discard the others."""
	for (i,mask) in enumerate(masks):
		labelled_mask, num_components = measurements.label(mask)
		areas = []
		if num_components > 1:
			for k in range(1, num_components+1):
				area = (labelled_mask==k).sum()
				areas.append(area)
			n = np.argmax(areas)+1
			masks[i,:,:] = labelled_mask==n
	return masks

def clean_up_pred_masks(masks, sigma=2):
	"""Smooths each mask by applying a Gaussian filter with std. dev. sigma, and subsequently
	removes smaller blobs from the mask using the function remove_unconnected_components"""
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
	"""Finds contour around mask in each box in rects. Returns a list containing (N,2) numpy arrays.
	Depending on the quality of the mask, no valid contours might be found. In this case, the box in
	question is skipped. A list of indices is returned of all boxes that contain a valid contour."""
	contours = []
	idxs = []
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
			idxs.append(i)
		except IndexError:
			continue
	return contours, idxs

def find_centroids(rects, masks):
	"""Computes the centroids of each mask in masks. Returns an (N,2) numpy array, where each slice
	[i,:] is a point (x,y)."""
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

def remove_shifted_centroids(centroids, contours):
	new_contours = []
	idxs = []
	for (k, cnt) in enumerate(contours):
		center = Point((centroids[k,1], centroids[k,0]))
		poly = Polygon([(cnt[j,0], cnt[j,1]) for j in range(cnt.shape[0])])
		if poly.contains(center):
			new_contours.append(cnt)
			idxs.append(k)
	return centroids[idxs], new_contours, idxs

def remove_duplicates(center_centroids, center_contours, other_centroids, shift, overlap_distance=-1):
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

# ======================================= Experimental ===========================================
def create_big_mask(c_im, rects, masks):
	"""Creates a binary image (mask) of size c_im.shape that incorporates each mask in masks. Only use for
	testing purposes, since overlapping masks are merged."""
	big_mask = np.zeros((c_im.shape[0], c_im.shape[1]))
	for i in range(rects.shape[0]):
		x, y, w, h = rects[i,:]
		mask = masks[i,...]
		# reshaped_mask = cv2.resize(mask.astype(np.uint8), (w,h))
		big_mask[y:y+h, x:x+w] = np.logical_or(mask, big_mask[y:y+h, x:x+w])
	return big_mask

def find_contours_big(big_mask):
	"""Finds the contours around each blob in big_mask. Returns a numpy array of shape (N, 2), where N
	is the number of contours. Note that if two crops are overlapping, we get one contour for both crops."""
	contours = measure.find_contours(big_mask, 0.5)
	return contours

def compute_mask_weirdness(big_mask, centroids):
	labelled_big_mask, num_components = measurements.label(big_mask)
	contours = []
	ws = []
	for i in range(1,num_components+1):
		isolated_mask = (labelled_big_mask == i).astype(np.uint8)
		cnt = measure.find_contours(isolated_mask, 0.5)
		contours.append(cnt[0])
		xs, ys = cnt[0][:,0], cnt[0][:,1]
		poly = Polygon([(xs[i], ys[i]) for i in range(len(xs))])
		L = poly.length
		A = poly.area
		ws.append(L)
		print(A, L, ws[-1])
	return labelled_big_mask, np.array(contours), ws

def triangulate(centroids):
	tri = Delaunay(centroids)
	return tri 					# to plot: ax.triplot(centroids[:,0], centroids[:,1], tri.simplices, color='w', lw=0.5)


if __name__ == "__main__":
	N = 50
	x = np.linspace(0,1,N)
	X, Y = np.meshgrid(x,x)
	M2 = (X-0.75)**2 + (Y-0.75)**2 < 0.1**2
	M1 = (X-0.25)**2 + (Y-0.3)**2 < 0.2**2
	M = np.logical_or(M1, M2)
	masks = np.zeros((2,N,N))
	masks[0,...] = M1
	masks[1,...] = M

	masks = remove_unconnected_components(masks)

	import matplotlib.pyplot as plt
	plt.imshow(masks[1,...])
	# for (i, cnt) in enumerate(cnts):
	# 	plt.plot(cnt[:,1], cnt[:,0], 'w', lw=0.5)
	# 	plt.text(cnt[:,1].mean(), cnt[:,0].mean(), '{:.3f}'.format(w[i]), ha='center')
	plt.show()
