# python predict_alibi.py -i dataset/test -m model

from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from alibi_detect.utils.saving import load_detector

import tensorflow as tf

from PIL import Image
from imutils import paths
import pandas as pd
import numpy as np
import imutils
import glob
import cv2
import os
import time

#----- Hyper-parameters --------------------------------------------------#
train_size = 64

testpath = "dataset/train" 

modelpath = 'models'
outputpath = 'results'
#----- Hyper-parameters --------------------------------------------------#

def img_to_np(image, resize = True):  
	image = cv2.resize(image, (train_size, train_size))
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image.astype("float") / 255.0
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

ng_cnt = ok_cnt = 0
imagePaths = sorted(list(paths.list_images(testpath)))
for i, fpath in enumerate(imagePaths):
	ins_time = time.time()
	image = cv2.imread(fpath)
	filename = fpath.split(os.path.sep)[-1]
	# print(filename)

	(H, W) = image.shape[:2]
	output = image.copy()

	# load image
	test = img_to_np(output)

	# test
	od = load_detector(modelpath)
	od.infer_threshold(test, threshold_perc=95.)

	preds = od.predict(test, 
		# outlier_type='instance', # instance' or 'feature'
		outlier_perc=100.,
		batch_size=int(1e10),
		return_instance_score=True,
		return_feature_score=True)

	result = int(preds['data']['is_outlier'])
	iscore = float(preds['data']['instance_score'])
	print(f'{filename}: {result} - {iscore:.5f}, {time.time()-ins_time:.5f}sec')
	
	text = f'is_outlier: {iscore:.5f}'
	if result == 1:
		cv2.rectangle(output, (0, 0), (W, H), (0, 0, 255), 5)
		cv2.putText(output, 'is_outlier', (15, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.5, (0, 0, 255),1)
		key = cv2.waitKey(1) & 0xFF
		ng_cnt += 1

	else:
		cv2.rectangle(output, (0, 0), (W, H), (0, 255, 0), 5)
		cv2.putText(output, 'ok', (15, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.5, (0, 255, 0), 1)
		key = cv2.waitKey(1) & 0xFF
		ok_cnt += 1
	
	os.makedirs(outputpath, exist_ok=True)
	cv2.imwrite(f'{outputpath}/{filename}.jpg', output) 
	
	output = imutils.resize(output, width = 800)
	cv2.imshow("Output", output)

	if key == ord("q"):
		break

print('\n\n\n[REULT]')
print(f'ok_cnt : {ok_cnt} ({ok_cnt/(ok_cnt+ng_cnt)*100:.2f}%) / ng_cnt: {ng_cnt} ({ng_cnt/(ok_cnt+ng_cnt)*100:.2f}%) / total: {ok_cnt+ng_cnt}')
