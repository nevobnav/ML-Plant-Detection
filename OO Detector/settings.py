#!/usr/bin/python2.7

"""
This script contains classes which store detection settings for
various different crops and model architectures.
"""

class DetectionSettings(object):
	"""Base detection settings class. 

	A child should implement the method parameters(), in which all 
	necessary model parameters are stored.

	Methods
	-------
	parameters()
		Function that contains all default settings. It should define
		the following attributes:
		* box_size : int
			Size of the bounding boxes in terms of RGB-pixels. Its value 
			depends on the size of the crops. For broccoli, good default
			values are 50 or 55. For lettuce, we recommend smaller boxes
			such that there is less overlap; use 40 or 45.
		* block_size : int
			Size of the blocks in which the RGB tif is divided, in terms
			of RGB-pixels. The bigger the blocks, the more data is loaded
			in memory at once. For testing purposes, we recommend a value
			of 500. On a GPU implementation, higher values like 2000 or
			3000 can be used.
		* block_overlap : int
			Width of the overlap region between two adjacent blocks. Keep
			at 3*box_size.
		* overlap_threshold : float (in [0,1])
			Minimum intersection over union (IoU) overlap between two
			boxes in order for non-max-suppression to trigger. For 
			broccoli detection, a good default setting is 0.4. For 
			lettuce detection, use a higher value if crops are closer
			together, we recommend around 0.6.
		* crop_size_threshold : float (in [0,1])
			Minimum percentage of bounding box that should be filled 
			with crop mask. If it is lower, the crop is discarded. A 
			good default value is 0.1. Higher value means more crops 
			are discarded
		* centroid_distance : int
			Minimum distance in RGB pixels between two centroids of 
			duplicate crops. If their distance is smaller, one of the crops
			will be removed. For broccoli, a good default value is half
			the box_size.
		* sigma : float (>0)
			Smoothing parameter in the region proposal step. Lower means 
			more candidate bounding boxes are detected. For broccoli, good 
			values are in the interval [3,6]. For lettuce we recommend 
			slightly lower values.
		* model_path : str (path object)
			Path to stored network. Should not have a suffix. If the network 
			is stored in the file /some/path/network.h5, set model_path
			equal to '/some/path/network'.
	"""

	def __init__(self, **kwargs):
		"""An attribute can be altered by adding it as a keyword argument
		to the constructor."""
		self.parameters()
		self.allowed_keys = set(self.__dict__)
		self.__dict__.update((key, val) for key, val in kwargs.items() if key in self.allowed_keys)

	def __str__(self):
		s= 'Settings(\n'+' '*5
		for key in self.__dict__:
			if key != 'allowed_keys':
				s+='{:<20} : {:<20}\n'.format(key, self.__dict__[key])+' '*5
		s = s+'   )'
		return s

	def parameters(self):
		raise NotImplementedError

class BroccoliUnifiedSettings(DetectionSettings):
	"""Settings to use for detecting broccoli crops using 
	a unified network"""

	def parameters(self):
		self.overlap_threshold = 0.4
		self.crop_size_threshold = 0.1					
		self.centroid_distance = 25
		self.box_size = 55
		self.sigma = 5
		self.model_path = '../Unified CNNs/Broccoli v4'
		self.block_size = 500
		self.block_overlap = 3*self.box_size
