# USAGE
# python test_eff111.py -m model/epoch_95.hdf5 -l model/model.pickle -i 02_image_test

# import the necessary packages
from keras.preprocessing.image import img_to_array
from pyimagesearch.preprocessing import PatchPreprocessor
import efficientnet.keras as efn
from keras.models import load_model
import random
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="model path")
ap.add_argument("-l", "--labelbin", required=True, help="label path")
ap.add_argument("-i", "--image", required=True,	help="test image path")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["image"])))
random.seed(42)

for (num, imagePath) in enumerate(imagePaths):
	# print(imagePath)
	file_name = imagePath.split(os.path.sep)[-1]
	orilabel = imagePath.split(os.path.sep)[1]
	# print(orilabel)

	image = cv2.imread(imagePath)
	(H, W) = image.shape[:2]
	output = image.copy()

	# pre-process the image for classification
	crop = cv2.resize(image, (224, 224))
	crop = crop.astype("float") / 255.0
	crop = img_to_array(crop)
	crop = np.expand_dims(crop, axis=0)

	# classify the input image
	proba = model.predict(crop)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	showcolor = (0,255,0) if 'ok' in label.lower() else (0,0,255)
	showlabel = "{} = {} : {:.2f}%".format(orilabel, label, proba[idx] * 100)
	print(showlabel)

	cv2.rectangle(output, (0, 0), (W, H), showcolor, 4)
	cv2.putText(output, showlabel, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, showcolor, 2)

	os.makedirs(f"OUTPUT/{orilabel}/O_{label}", exist_ok=True)
	os.makedirs(f"OUTPUT/{orilabel}/X_{label}", exist_ok=True)
	if orilabel == label:
		cv2.imwrite(f"OUTPUT/{orilabel}/O_{label}/{file_name}", output)
		key = cv2.waitKey(1) & 0xFF
	else:
		cv2.imwrite(f"OUTPUT/{orilabel}/X_{label}/{file_name}", output)
		key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break	
		
	output = imutils.resize(output, width=1024)
	cv2.imshow("Output", output)
