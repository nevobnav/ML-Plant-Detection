# ML-Plant-Detection

## Contents
This branch contains the following scripts:

#### dem_functions.py
Contains functions that allows one to communicate between RGB tif and DEM tif.
#### processing.py
Contains all pre- and post-processing functions.
#### settings.py
Contains classes that encapsulate the parameters sets for different crops.
#### main.py
Contains complete model and output functions.

It also contains the two folders Detection CNNs and Masking CNNs, which contain the neural networks used for classification and masking respectively. The models are saved in the .h5 format, and thus contain both the network structure and the model weights.

## Usage
In main.py, let the variables `img_path` and `dem_path` equal paths to a RGB tif and dem respectively. Set `crop` equal to the type of crop present in the field. Currently, only broccoli and lettuce are supported. To change one of the parameters, one does not have to edit settings.py. Instead, add the altered parameter value as an optional keyword argument to the function `get_settings`. For example: `params = get_settings(crop, box_size=50)`. Run the model by executing the script main.py. The output will be written to 3 different shapefiles; BLOCK_LINES.shp, CONTOURS.shp and POINTS.shp. These shapefiles are saved in the folder ./PLANT COUNT - img_name, where img_name is the name of the input tif. The file BLOCK_LINES.shp contains polygons representing the blocks in which the input tif is divided, CONTOURS.shp contains polygons representing the outlines of detected crops and POINTS.shp contains points representing their centroids.