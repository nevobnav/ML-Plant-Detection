def get_settings(crop, **kwargs):
	if crop.lower() == 'broccoli':
		return BroccoliSettings(**kwargs).params
	elif crop.lower() == 'lettuce':
		return LettuceSettings(**kwargs).params
	elif crop.lower() == 'ijsberg':
		return IJsbergSettings(**kwargs).params

class BroccoliSettings(object):
	def __init__(self, **kwargs):
		overlap_threshold 	= 0.4					# for broccoli, use 0.4, for lettuce, use 0.7
		crop_size_threshold = 0.05					# 0.0 - 0.1, percentage of bounding box that should be filled with crop
		center_distance 	= 0.05					# 0.0 - 0.5, relative distance a mask centroid can be from box center before the box is recentered.
		overlap_distance    = 25					# minimum distance in pixels between two centroids
		box_size = 55								# For broccoli, use 55-60, for lettuce, use 45-50. Must be even if >64
		sigma = 5									# lower means more candidate bounding boxes are detected, good range is 2.5 - 7.5
		filter_masks 	= True
		recenter 		= True
		filter_disjoint = True
		c_box = (60, 60)
		h_box = (20, 20)
		box_model_name  = 'Detection CNNs/CROP7+SAHN4_t5.h5'					# Broccoli, use t5
		mask_model_name = 'Masking CNNs/broccoli_masker_500.h5'							# use masker_500
		block_size = 500
		block_overlap = int(1.5*box_size)

		self.params = {
				  'overlap_threshold':overlap_threshold,
				  'crop_size_threshold':crop_size_threshold,
				  'center_distance':center_distance,
				  'overlap_distance':overlap_distance,
				  'box_size':box_size,
				  'sigma':sigma,
				  'filter_masks':filter_masks,
				  'recenter':recenter,
				  'filter_disjoint':filter_disjoint,
				  'c_box':c_box,
				  'h_box':h_box,
				  'detection_model_path':box_model_name,
				  'masking_model_path':mask_model_name,
				  'block_size':block_size,
				  'block_overlap':block_overlap}

		for kwarg in kwargs.keys():
			self.params[kwarg] = kwargs[kwarg]


class LettuceSettings(object):
	def __init__(self, **kwargs):
		overlap_threshold 	= 0.6					# lower is stricter
		crop_size_threshold = 0.05					# 0.0 - 0.1, percentage of bounding box that should be filled with crop
		center_distance 	= 0.05					# 0.0 - 0.5, relative distance a mask centroid can be from box center before the box is recentered.
		overlap_distance    = 20					# minimum distance in pixels between two centroids
		box_size = 50								# For broccoli, use 55-60, for lettuce, use 45-50. Must be even if >64
		sigma = 4									# lower means more candidate bounding boxes are detected, good range is 2.5 - 7.5
		filter_masks 	= True
		recenter 		= True
		filter_disjoint = True
		c_box = (60, 60)
		h_box = (20, 20)
		box_model_name  = 'Detection CNNs/lettuce_v7_C49.h5'
		mask_model_name = 'Masking CNNs/lettuce_masker_350.h5'
		block_size = 500
		block_overlap = int(1.5*box_size)

		self.params = {
				  'overlap_threshold':overlap_threshold,
				  'crop_size_threshold':crop_size_threshold,
				  'center_distance':center_distance,
				  'overlap_distance':overlap_distance,
				  'box_size':box_size,
				  'sigma':sigma,
				  'filter_masks':filter_masks,
				  'recenter':recenter,
				  'filter_disjoint':filter_disjoint,
				  'c_box':c_box,
				  'h_box':h_box,
				  'detection_model_path':box_model_name,
				  'masking_model_path':mask_model_name,
				  'block_size':block_size,
				  'block_overlap':block_overlap}

		for kwarg in kwargs.keys():
			self.params[kwarg] = kwargs[kwarg]

class IJsbergSettings(LettuceSettings):
	def __init__(self, **kwargs):
		LettuceSettings.__init__(self, **kwargs)
		self.params['box_size'] = 65
		self.params['overlap_threshold'] = 0.5

if __name__ == "__main__":
	params = get_settings('broccoli', block_size=1200)
	print(params)