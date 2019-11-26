#!/usr/bin/python3.6

"""
Script containing functions for loading, saving and extracting information
such as input shapes from a network. Also contains the class NetworkTrainer,
which implements a straightforward way of initializing and training a crop
detection/masking network.
"""

import os

from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as prep

__author__ = "Duncan den Bakker"

def load_network(path, platform):
	"""Loads a neural network.

	If platform=='linux', the model is loaded from a .h5 file using
	keras.models.load_model(path). If platform=='windows', the model
	architecture is first loaded from a .json, after which the
	weights are loaded separately.

	Arguments
	---------
	path : str (path object)
		Path to folder containing the files NETWORK.h5, STRUCTURE.h5 and
		WEIGHTS.h5.
	platform : str
		OS, either linux or windows.
	"""

	if platform=='linux':
		network = models.load_model(path+'/NETWORK.h5')

	elif platform == 'windows':
		from keras.utils import CustomObjectScope
		from keras.initializers import glorot_uniform
		with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
			with open(path+'/STRUCTURE.json', 'r') as f:
				json_string = f.read()
				network = models.model_from_json(json_string)
				network.load_weights(path+'/WEIGHTS.h5')

	return network

def save_network(network, path):
	"""Saves a trained network.

	Saves both the entire network structure and weights in one .h5 file, as
	well as a separate .json structure file and .h5 weight file. Only the latter
	can be loaded on windows due to a bug in keras. A folder is created at path.
	The files NETWORK.h5, STRUCTURE.json and WEIGHTS.h5 are saved to this folder.

	Arguments
	---------
	network : keras model
		Network to be saved.
	path : str (path object)
		Path to folder in which files must be saved.
	"""

	if not os.path.exists(path):
		os.makedirs(path)
	json_string = network.to_json()
	with open(path+'/STRUCTURE.json', 'w') as f:
		f.write(json_string)
	network.save_weights(path+'/WEIGHTS.h5')
	network.save(path+'/NETWORK.h5')
	print('Succesfully saved network files to folder {}'.format(path))

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

class NetworkTrainer(object):
	"""Class that allows for easy model initialization and training.

	Attributes
	----------
	num_classes : int
		Number of different classes that the network must be able to
		detect.
	rgb_size : 2-tuple
		Spatial dimensions of RGB input of network. Keep at (64,64) to
		maintain compatibility in FCN network.
	dem_size : 2-tuple
		Spatial dimensions of DEM input of network. Default is (16,16).
		In general there is no need to change this.
	network : keras model
		Detection network object.
	data_generator : iterator object
		Object yielding input-output pairs to train the network on.
		Must be initialized by the user with the method set_training_data.

	Methods
	-------
	init_network_structure():
		Defines the network architecture.
	compile(learning_rate=1e-4, mask_loss_weight=0.75):
		Compiles self.network with Adam optimizer.
	set_training_data(data_path):
		Initializes self.data_generator with data from data_path.
	train(epochs, steps_per_epoch):
		Trains network.
	save(path):
		Saves network.
	"""

	def __init__(self, num_classes):
		"""
		Arguments
		---------
		num_classes : int (>=2)
			Number of different classes network should be able to distinguish.
		"""

		self.num_classes = num_classes
		self.rgb_size = (64,64)
		self.dem_size = (16,16)

		self.network = self.init_network_structure()
		self.data_generator = None

	def init_network_structure(self):
		"""Network structure definition.

		Defines a multi-input, multi-output Keras model architecture consisting
		of 4 different parts. See documentation for details.
		"""

		# Color image convolutional network
		input_RGB = layers.Input(shape=(self.rgb_size[0], self.rgb_size[1], 3))
		x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_RGB)
		x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
		x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2,2))(x)
		x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
		x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
		c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2,2))(c4)
		x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
		x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
		x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
		c7 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2,2))(c7)
		c_out = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
		x = layers.Flatten()(x)

		# Height image convolutional network
		input_DEM = layers.Input(shape=(self.dem_size[0], self.dem_size[1], 1))
		y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_DEM)
		y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
		y = layers.MaxPooling2D((2,2))(y)
		y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
		y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
		y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
		y = layers.MaxPooling2D((2,2))(y)
		y = layers.Conv2D(256, (3,3), activation='relu', padding='same')(y)
		y = layers.Conv2D(256, (3,3), activation='relu', padding='same')(y)
		y = layers.Conv2D(256, (3,3), activation='relu', padding='same')(y)
		h_out = layers.Conv2D(512, (3,3), activation='relu', padding='same')(y)
		y = layers.Flatten()(h_out)

		# Combine Color and Height networks into FC layers
		combined = layers.concatenate([x, y])
		z = layers.Dense(256, activation='relu')(combined)
		z = layers.Dense(128,  activation='relu')(z)
		z = layers.Dense(64,  activation='relu')(z)
		z = layers.Dense(32,  activation='relu')(z)
		output_class = layers.Dense(self.num_classes, activation='softmax', name='output_class')(z)

		# FCN Masking part
		f2 = layers.Conv2D(filters=1, kernel_size=1)(c_out)
		f2 = layers.Conv2DTranspose(filters=c7.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f2)
		f3 = layers.Conv2D(filters=1, kernel_size=1)(c7)
		f3 = layers.Add()([f2, f3])
		f3 = layers.Conv2D(filters=1, kernel_size=1)(f3)
		f3 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f3)
		f3 = layers.Add()([f3, c4])
		output_mask = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='output_mask')(f3)

		network = keras.Model(inputs=[input_RGB, input_DEM], outputs=[output_class, output_mask])

		return network

	def compile(self, learning_rate=1e-4, mask_loss_weight=0.75):
		"""Compile model with Adam optimizer.

		Use categorical crossentropy loss function for the class output,
		and mean squared error for the mask output. The total loss is the
		weighted sum class_loss + mask_loss_weight*mask_loss. This method
		has no return value, it compiles the attribute self.network.

		Arguments
		---------
		learning_rate : float
			Learning rate of Adam optimizer. We observed the keras default of
			1e-3 is too big. The default value is 1e-4.
		mask_loss_weight : float
			Weight of mask loss function compared to class loss function.
		"""

		losses = {"output_class": 'categorical_crossentropy', "output_mask":'mse'}
		loss_weights = {"output_class": 1.0, "output_mask": mask_loss_weight}
		metrics = {'output_class':'accuracy', 'output_mask':'binary_accuracy'}
		opt = keras.optimizers.Adam(learning_rate=learning_rate)
		self.network.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)

	def set_training_data(self, data_path, batch_size=64):
		"""Initializes a generator object.

		This method has no return value, it assigns an iterator object
		to the attribute set_training_data.

		Arguments
		---------
		data_path : str (path object)
			Path to a folder with the following structure:
				data_path
				|-- RGB
				|	|-- Class 0 - Class_0_Name
				|	|-- Class 1 - Class_1_Name
				|	:
				|	|-- Class N - Class_N_Name
				|-- DEM
				|	|-- Class 0 - Class_0_Name
				|	|-- Class 1 - Class_1_Name
				|	:
				|	|-- Class N - Class_N_Name
				|-- MSK
				|	|-- Class 0 - Class_0_Name
				|	|-- Class 1 - Class_1_Name
				|	:
				|	|-- Class N - Class_N_Name
			Class 0 is usually reserved for background.
		batch_size : int, optional
			Size of batch for which the cumulative gradient is calculated.
			Higher value means more compations are performed. Default value
			of 64 need not be changed.

		Raises
		------
		FileNotFoundError : if data_path does not exist in filesystem.
		IOError : if at least one of the folders RGB, DEM, MSK has not
			been found in data_path.
		IndexError : if number of classes in the folders RGB, DEM, MSK
			is not equal to self.num_classes.
		"""

		if not os.path.exists(data_path):
			raise FileNotFoundError('Path "{}" does not exist'.format(data_path))

		if not os.path.exists(data_path+'/RGB/') \
			or not os.path.exists(data_path+'/DEM/') \
			or not os.path.exists(data_path+'/MSK/'):
			raise IOError('One of more of the folders {}, {}, {} have not been found in {}.\n\
				Either the path is wrong or the dataset has a wrong structure.'.format('/RGB/', '/DEM/', '/MSK/', data_path))

		if len(os.listdir(data_path+'/RGB/')) != self.num_classes:
			raise IndexError('Number of classes in {} not equal to self.num_classes'.format(data_path+'/RGB/'))

		rgb_size, dem_size = get_input_sizes(self.network)
		gen_object = prep.image.ImageDataGenerator()

		def DataIterator(generator, dir_rgb, dir_dem, dir_msk, seed=0):
			rgb_flow = generator.flow_from_directory(dir_rgb, target_size=rgb_size, class_mode='categorical',
				batch_size=batch_size, shuffle=True, seed=seed, color_mode='rgb')
			dem_flow = generator.flow_from_directory(dir_dem, target_size=dem_size, class_mode='categorical',
				batch_size=batch_size, shuffle=True, seed=seed, color_mode='grayscale')
			msk_flow = generator.flow_from_directory(dir_msk, target_size=rgb_size, class_mode='categorical',
				batch_size=batch_size, shuffle=True, seed=seed, color_mode='grayscale')
			while True:
				rgb_array = rgb_flow.next()
				dem_array = dem_flow.next()
				msk_array = msk_flow.next()
				yield [rgb_array[0], dem_array[0]], [rgb_array[1], msk_array[0]]

		self.data_generator = DataIterator(gen_object, data_path+'/RGB/', data_path+'/DEM/', data_path+'/MSK/')

	def train(self, epochs, steps_per_epoch):
		"""Trains the compiled model.

		A dataset must first be specified using the method set_training_data.

		Arguments
		---------
		epochs : int
			Number of training epochs.
		steps_per_epoch : int
			Number of steps per epoch.

		Raises
		------
		ValueError : if no dataset has been set.
		"""

		if self.data_generator == None:
			raise ValueError('No training data specified. Use the method set_training_data.')
		self.network.fit_generator(self.data_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

	def save(self, path):
		"""Saves network using the function save_network."""
		save_network(self.network, path)

if __name__ == "__main__":
	data_path = '/home/duncan/Documents/VanBoven/DL Datasets/Unified Broccoli (NEW FORMAT)'

	Trainer = NetworkTrainer(2)
	Trainer.compile()
	print(Trainer.network.summary())
	# Trainer.set_training_data(data_path)
	# Trainer.train(1,1)
	# Trainer.save('./Test Network')
