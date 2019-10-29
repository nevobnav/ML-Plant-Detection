# ML-Plant-Detection

## Contents
This branch contains the following scripts:

#### tif_functions.py
Contains functions that allows one to communicate between RGB tif and DEM tif.
#### processing.py
Contains all pre- and post-processing functions.
#### settings.py
Contains classes that encapsulate the parameters sets for different crops.
#### main.py
Contains complete model and output functions.

It also contains the three folders Detection CNNs, Masking CNNs and Field Shapefiles. The first two folders contain the neural networks used for classification and masking respectively. The models are saved in the .h5 format, and thus contain both the network structure and the model weights. The folder Field Shapefiles contains shapefiles with polygons that represent the part of the plot which actually contains crops. Such shapefiles are currently available for a number of different plots, and are made manually. Ideally, these shapefiles should be generated automatically by some algorithm.

## Usage
In main.py, let the variables `img_path` and `dem_path` equal paths to a RGB tif and dem respectively. If there is one, let `clp_path` equal a path to a shapefile which contains polygons that represent the crop fields. If there is no such shapefile, set it to `None`. Set `crop` equal to the type of crop present in the field. Currently, only broccoli and lettuce are supported. To change one of the parameters, one does not have to edit settings.py. Instead, add the altered parameter value as an optional keyword argument to the function `get_settings`. For example: `params = get_settings(crop, box_size=50)`. Run the model by executing the script main.py. The output will be written to 3 different shapefiles; BLOCK_LINES.shp, CONTOURS.shp and POINTS.shp. These shapefiles are saved in the folder ./PLANT COUNT - img_name, where img_name is the name of the input tif. The file BLOCK_LINES.shp contains polygons representing the blocks in which the input tif is divided, CONTOURS.shp contains polygons representing the outlines of detected crops and POINTS.shp contains points representing their centroids.

## A Note on Platforms
The algorithm is written on the Ubuntu (Linux) platform, but is intended to be used on all platforms (Windows, Mac). In main.py, set the variable `platform` equal to the OS on which the algorithm is being run. This is necessary because file paths have a different structure on different platforms (this should be fixed by using the module `pathlib` in a future version). Furthermore, the models are trained and saved on Linux. Due to a bug in Keras, .h5 files created on Linux can not be loaded on Windows using the `load_model` method from `keras.models`. Instead, the model structure and weights should be saved separately to a .json and .h5 file respectively. These files can then by imported on Windows using a workaround. A small Python script `model_converter.py` is added to both CNN folders which converts a complete model .h5 file to a .json structure files and .h5 weight file.

## Current Problems and Planned Changes

### File Path Management
To improve cross-compatibility between Linux and Windows (and OSX), file paths should be handled by the module `pathlib` instead of dealing with strings.

### Field Edge False Positives
Faulty crops are detected near the edge of the clipped field. It is to be expected that centers are detected near these edges, since the no-data value defaults to black. However, the CNN seems to label these locations as crops. Possible fixes are re-training the model with some of these edge-cases, or maybe also clipping the DEM. This phenomenon only seems to occur at certain borders. 

### Big Sizes Difference
Since the box size is fixed, the model tends to miss crops which are very small relative to the box. This can be fixed by placing boxes of various sizes at the same location. If a crop is small, it will probably be detected in a small box, but not a big one. If a crop is detected in every box, keep only the biggest. A drawback of this method is that the detection CNN will have to process much more candidates, significantly impacting computation time. 

## Other Notes
The model result for the plot c01_verdonk-Wever west-201907240724-GR shows two mysterious strips in which no crops have been detected. The strips are not straight, but cross the entire plot. It might have something to do with the crop size. These strips have not been spotted in other plots, so we suspect it has to do with the detection network.