#!/usr/bin/python3.6

import numpy as np
# from pre_proc import mask_filter
import cv2
import tensorflow.keras.models as models
from scipy.ndimage import filters
from scipy.ndimage import measurements
from skimage import measure
import skimage.color

from shapely.geometry import Polygon
from scipy.spatial import Delaunay
from skimage.feature import peak_local_max

# ========================================= Region Proposal =============================================

def window_hotspots_centers(im, sigma=3.6, padding=0, m=2, top_left=0):
	"""RoI generator based on local minima of (u+v)-channel."""
	if padding > 0:
		im = im[padding:-padding, padding:-padding, :]
	im_yuv = skimage.color.rgb2yuv(im)
	im_filtered = im_yuv[:,:,1] + im_yuv[:,:,2]
	im_filtered = filters.minimum_filter(im_filtered, m)
	im_filtered = filters.gaussian_filter(im_filtered, sigma)
	mask = peak_local_max(-im_filtered, indices=False)#, exclude_border=False)
	mask = np.logical_and(mask, im_filtered!=0)
	if top_left > 0:
		other = np.ones(mask.shape, dtype=int)
		other[:top_left, :top_left] = 0
		mask = np.logical_and(mask, other)
	coords = np.argwhere(mask==1)
	return coords+padding

# =========================================== Box filters ===============================================

def sort_into_classes(rects, predictions, weeds=False):
	"""Sort bounding boxes into the class broccoli, background (."""
	num_candidates = predictions.shape[0]
	back_box_indeces, broc_box_indeces, weed_box_indeces = [], [], []
	back_rects, broc_rects, = [], []
	back_prob,  broc_prob,  = [], []
	if weeds:
		weed_rects, weed_prob = [], []

	for i in range(num_candidates):
		pred_index = np.argmax(predictions[i,:])		# prediction index corresponds to label name
		if pred_index == 0:								# background
			back_box_indeces.append(i)					# store index belonging to predicted broccoli
			back_rects.append(rects[i,:])
			back_prob.append(predictions[i,pred_index])
		elif pred_index == 1:							# broccoli
			broc_box_indeces.append(i)					# store index belonging to predicted broccoli
			broc_rects.append(rects[i,:])
			broc_prob.append(predictions[i,pred_index])
		elif pred_index == 2 and weeds:							# weed
			weed_box_indeces.append(i)					# store index belonging to predicted weed
			weed_rects.append(rects[i,:])
			weed_prob.append(predictions[i,pred_index])

	return (np.array(back_rects), np.array(back_prob)), \
		   (np.array(broc_rects), np.array(broc_prob))
		   # (np.array(weed_rects), np.array(weed_prob))

def non_max_suppression(boxes, probs=[], t=0.2):
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
	if len(probs) > 0:
		return boxes[pick], probs[pick]					# list as index only works with numpy arrays
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

def discard_empty(rects, prob, masks, t=0.01):
	"""Discards rectangles which are nearly empty. The parameter t determines the minimum amount of the
	box that should be filled as a fraction of the total area."""
	crop_areas = np.sum(np.sum(masks, axis=2), axis=1)
	is_almost_empty = crop_areas < t*rects[:,2]*rects[:,3]
	filtered_rects = np.array([elt for (i,elt) in enumerate(rects) if not is_almost_empty[i]])
	filtered_prob  = np.array([elt for (i,elt) in enumerate(prob)  if not is_almost_empty[i]])
	filtered_masks = np.array([elt for (i,elt) in enumerate(masks) if not is_almost_empty[i]])
	return filtered_rects, filtered_prob, filtered_masks

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
		if num_components >= 1:
			for k in range(1, num_components+1):
				area = (labelled_mask==k).sum()
				areas.append(area)
			n = np.argmax(areas)+1
			masks[i,:,:] = labelled_mask==n
	return masks

def find_contours(rects, masks):
	"""Finds contour around mask in each box in rects. Returns a list containing (N,2) numpy arrays."""
	contours = []
	for (i, rect) in enumerate(rects):
		x, y, w, h = rect
		rel_cnt = measure.find_contours(masks[i,...], 0.5)[0]
		rel_cnt[:,0] += y
		rel_cnt[:,1] += x
		contours.append(rel_cnt)
	return contours

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
		xc = np.sum(xs*mask)*box_size/mask_width
		yc = np.sum(ys*mask.T)*box_size/mask_height
		centroids[i,:] = [x+xc/area, y+yc/area]
	return centroids

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

	cnts = find_contours(M)
	cnts = np.dstack((cnt for cnt in cnts))
	print(cnts.shape)

	import matplotlib.pyplot as plt
	plt.imshow(mask)
	for (i, cnt) in enumerate(cnts):
		plt.plot(cnt[:,1], cnt[:,0], 'w', lw=0.5)
		plt.text(cnt[:,1].mean(), cnt[:,0].mean(), '{:.3f}'.format(w[i]), ha='center')
	plt.show()
		