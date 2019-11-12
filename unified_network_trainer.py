#!/usr/bin/python3.6

from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as prep
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow.keras.backend as K

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# import json

batch_size = 32

#======================================== Loading/Saving Model ========================================
def load_separate(path):
	with open(path+'.json', 'r') as f:
		json_string = f.read()
	new_model = models.model_from_json(json_string)
	new_model.load_weights(path+'_weights.h5')
	return new_model

def save_separate(model, out_name):
	json_string = model.to_json()
	with open(out_name+'.json', 'w') as f:
		f.write(json_string)
	model.save_weights(out_name+'_weights.h5')

#======================================= Custom Loss functions ========================================
def jaccard_loss(y_true, y_pred, smooth=100):
    """Jaccard/IoU loss function."""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

#======================================== Model Architectures =========================================
def create_network_base(num_classes=2):
	c_size = (64,64)
	h_size = (20,20)

	# Color image convolutional network
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
	x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_RGB)
	x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(x)
	x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
	c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(c4)
	x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(c7)
	x = layers.Flatten()(x)

	# Height image convolutional network
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y = layers.Conv2D(16, (3,3), activation='relu')(input_H)
	y = layers.Conv2D(16, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)			#added
	y = layers.Flatten()(y)

	# Combine Color and Height networks into FC layers
	combined = layers.concatenate([x, y])
	z = layers.Dense(128, activation='relu')(combined)
	z = layers.Dense(32,  activation='relu')(z)
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	# FCN Masking part
	fcn1 = layers.Conv2D(filters=1, kernel_size=1, name='fcn1')(c7)
	fcn2 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', name='fcn2')(fcn1)
	fcn3 = layers.Add()([fcn2, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(fcn3)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	return model

def create_network_v2(num_classes=2):
	"""
	Differences compared to base:
		- Added more layers and more feature maps in RBG and H networks.
		- Increased number of fully connected layers in classification networks
	"""
	c_size = (64,64)
	h_size = (20,20)

	# Color image convolutional network
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
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
	x = layers.Flatten()(x)

	# Height image convolutional network
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y = layers.Conv2D(32, (3,3), activation='relu')(input_H)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)			#added
	y = layers.Flatten()(y)

	# Combine Color and Height networks into FC layers
	combined = layers.concatenate([x, y])
	z_top = layers.Dense(256, activation='relu')(combined)
	z = layers.Dense(128,  activation='relu')(z_top)
	z = layers.Dense(64,  activation='relu')(z)
	z = layers.Dense(32,  activation='relu')(z)
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	# FCN Masking part
	f = layers.Conv2D(filters=1, kernel_size=1)(c7)
	f = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', name='fcn2')(f)
	f = layers.Add()([f, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(f)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	return model

def create_network_v3(num_classes=2):
	"""
	Differences compared to v2:
		- Added another Conv layer in RGB part that is connected to FCN
	"""
	c_size = (64,64)
	h_size = (20,20)

	# Color image convolutional network
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
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
	c9 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
	x = layers.Flatten()(x)

	# Height image convolutional network
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y = layers.Conv2D(32, (3,3), activation='relu')(input_H)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)			#added
	y = layers.Flatten()(y)

	# Combine Color and Height networks into FC layers
	combined = layers.concatenate([x, y])
	z = layers.Dense(256, activation='relu')(combined)
	z = layers.Dense(128,  activation='relu')(z)
	z = layers.Dense(64,  activation='relu')(z)
	z = layers.Dense(32,  activation='relu')(z)
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	# FCN Masking part
	f1 = layers.Conv2D(filters=1, kernel_size=1)(c9)
	f1 = layers.Conv2DTranspose(filters=c7.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f1)
	f2 = layers.Conv2D(filters=1, kernel_size=1)(c7)
	f2 = layers.Add()([f2, f1])
	f2 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f2)
	f2 = layers.Add()([f2, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(f2)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	return model

def create_network_v4(num_classes=2):
	"""
	Differences compared to v3:
		- Reduced Height input size from (20,20) to (16,16) to make it compatible with mask network.
		- Increased height network and added its output to masking step.
	"""
	c_size = (64,64)
	h_size = (16,16)

	# Color image convolutional network
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
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
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_H)
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
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	# FCN Masking part
	f1 = layers.Conv2D(filters=1, kernel_size=1)(h_out)
	f1 = layers.Conv2DTranspose(filters=c_out.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f1)
	f2 = layers.Conv2D(filters=1, kernel_size=1)(c_out)
	f2 = layers.Add()([f1, f2])
	f2 = layers.Conv2DTranspose(filters=c7.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f2)
	f3 = layers.Conv2D(filters=1, kernel_size=1)(c7)
	f3 = layers.Add()([f2, f3])
	f3 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same')(f3)
	f3 = layers.Add()([f3, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(f3)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	return model

#========================================== Initialization ============================================
def compile_model(model, learning_rate=1e-4, mask_loss_weight=0.5):
	"""Compile model with adam optimizer with alpha=learning_rate. The parameter
	mask_loss_weight determines how 'important' the mask_output_loss is compared to the
	class_output_loss, with 1 denoting a 50/50 split."""
	# losses = {"class_output": 'categorical_crossentropy', "mask_output":'mse'}
	losses = {"class_output": 'categorical_crossentropy', "mask_output":'mean_squared_error'}
	loss_weights = {"class_output": 1.0, "mask_output": mask_loss_weight}
	metrics = {'class_output':'accuracy', 'mask_output':'binary_accuracy'}
	opt = keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)

def init_data_generator(data_path, model):
	"""Initializes a generator object that generates input-output tuples of the form
	[RGB_image, H_image], [Label, Mask]. The path data_path should contain the following folders:
		- 'Training Data Color/'
		- 'Training Data Height/'
		- 'Training Data Mask/'
	The model that is to be trained should also be passed as an argument, such that the size of
	its input layers can be determined."""

	c_size = model.get_input_at(0)[0].get_shape().as_list()[1:3]
	h_size = model.get_input_at(0)[1].get_shape().as_list()[1:3]

	dir_color  = 'Training Data Color/'
	dir_height = 'Training Data Height/'
	dir_mask   = 'Training Data Mask/'
	if not os.path.exists(data_path+dir_color) or not os.path.exists(data_path+dir_height) or not os.path.exists(data_path+dir_mask):
		raise IOError('The folders {}, {}, {} have not been found in {}. \
			Either the path is wrong or the dataset has a wrong structure.'.format(dir_color, dir_height, dir_mask, data_path))

	gen_object = prep.image.ImageDataGenerator()

	def Multi_Input_Output_Flow(generator, dir_c, dir_h, dir_m, seed=0):
		gen_color  = generator.flow_from_directory(dir_c, target_size=c_size, class_mode='categorical',
			batch_size=batch_size, shuffle=True, seed=seed, color_mode='rgb')
		gen_height = generator.flow_from_directory(dir_h, target_size=h_size, class_mode='categorical',
			batch_size=batch_size, shuffle=True, seed=seed, color_mode='grayscale')
		gen_mask   = generator.flow_from_directory(dir_m, target_size=c_size, class_mode='categorical',
			batch_size=batch_size, shuffle=True, seed=seed, color_mode='grayscale')
		while True:
			im_color  = gen_color.next()
			im_height = gen_height.next()
			im_mask   = gen_mask.next()
			yield [im_color[0], im_height[0]], [im_color[1], im_mask[0]]  				# [RGB, Height], [label, Mask]

	return Multi_Input_Output_Flow(gen_object, data_path+dir_color, data_path+dir_height, data_path+dir_mask)

#========================================= Visualize Result ===========================================
def show_predictions(k, gen, model, class_names=['Background', 'Broccoli']):
	"""Debugging method to show k (<=batch_size) random inputs and its corresponding model output."""
	[color_ims, height_ims], [labels, masks] = next(gen)
	[pred_labs, pred_masks] = model.predict([color_ims, height_ims])
	for i in range(k):
		f, axs = plt.subplots(2,3, figsize=(4,4))
		(ax1, ax2, ax3, ax4, ax5, ax6) = axs.flatten()
		ax1.imshow(color_ims[i,:,:,:].astype(np.uint8))
		ax4.imshow(pred_masks[i,:,:,0], cmap='gray')
		ax5.imshow(pred_masks[i,:,:,0]>0.5, cmap='gray')
		ax3.imshow(masks[i,:,:,0], cmap='gray')
		ax2.imshow(height_ims[i,:,:,0], cmap='Reds')
		for ax in axs.flatten():
			ax.axis('off')
		label_true = np.argmax(labels[i,...])
		label_pred = np.argmax(pred_labs[i,...])
		f.suptitle('{} : {}'.format(class_names[label_pred], class_names[label_true]), fontsize=8)
		ax1.set_title('Input (RGB)', fontsize=8)
		ax2.set_title('Input (H)', fontsize=8)
		ax3.set_title('Mask (true)', fontsize=8)
		ax4.set_title('Mask (soft)', fontsize=8)
		ax5.set_title('Mask (pred)', fontsize=8)
	plt.show()

#========================================= Visualize Result ===========================================

if __name__ == "__main__":
	master_dir = './testing/'

	# model = load_separate('Unified CNNs/broccoli_unified_v4')
	model = create_network_v4()

	gen = init_data_generator(master_dir, model)

	compile_model(model, mask_loss_weight=0.75)

	# model.fit_generator(gen, epochs=3, steps_per_epoch=20)
	# model.save('Unified CNNs/broccoli_unified_v4.h5')
	# save_separate(model, 'Unified CNNs/broccoli_unified_v4')

	# show_predictions(5, gen, model)
