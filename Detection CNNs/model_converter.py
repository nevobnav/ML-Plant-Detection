import tensorflow.keras.models as ker_models
import json

# path = 'Detection CNNs/CROP7+SAHN4_t5.h5'
path = 'lettuce_v6_C49.h5'
name = path.split('/')[-1].split('.')[0]
model = ker_models.load_model(path)

def save(model):
	json_string = model.to_json()
	with open(name+'.json', 'w') as f:
		f.write(json_string)
	model.save_weights(name+'_weights.h5')

def load(model_name):
	with open(model_name, 'r') as f:
		json_string = f.read()
	new_model = ker_models.model_from_json(json_string)
	new_model.load_weights(name+'_weights.h5')
	return new_model

save(model)
new_model = load(name+'.json')