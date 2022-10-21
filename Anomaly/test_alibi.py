from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from alibi_detect.utils.saving import load_detector
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to testpath")
ap.add_argument("-m", "--model", required=True,	help="path to modelpath")
args = vars(ap.parse_args())

def img_to_np(image, resize = True):  

	image = cv2.resize(image, (64, 64))
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image.astype("float") / 255.0
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.expand_dims(image, axis=0)

	return image

testpath = args["image"] + "/*.*"
modelpath = args["model"] +'/'

for i, fpath in enumerate(glob.glob(testpath)):
	image = cv2.imread(fpath)

	filename = fpath.split(os.path.sep)[-1]
	# print(filename)

	(H, W) = image.shape[:2]
	output = image.copy()

	test = img_to_np(output)

	od = load_detector(modelpath)
	od.infer_threshold(test, threshold_perc=80)

	preds = od.predict(test, 
				outlier_type='instance', # instance' or 'feature'
				return_instance_score=True,
				return_feature_score=True)

	result = preds['data']['is_outlier']
	print(f'filename: {filename}, result: {result}')

	if result == 1:
		cv2.rectangle(output, (0, 0), (W, H), (0, 0, 255), 10)
		cv2.putText(output, 'is_outlier', (50, 100),  cv2.FONT_HERSHEY_SIMPLEX,	3, (0, 0, 255),5)
		key = cv2.waitKey(1) & 0xFF

	else:
		cv2.rectangle(output, (0, 0), (W, H), (0, 255, 0), 10)
		cv2.putText(output, 'ok', (50, 100),  cv2.FONT_HERSHEY_SIMPLEX,	3, (0, 255, 0), 5)
		key = cv2.waitKey(0) & 0xFF
	
	os.makedirs('output', exist_ok=True)
	cv2.imwrite(f'output/{filename}.jpg', output) 
	
	output = imutils.resize(output, width = 800)
	cv2.imshow("Output", output)

	if key == ord("q"):
		break
