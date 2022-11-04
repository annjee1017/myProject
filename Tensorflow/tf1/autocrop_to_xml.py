# USAGE
# python save.py -i image_file

# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import time
from imutils import paths
import random
import os
import shutil

#######################################################################################################
min_confidence = 0.5
model_path = 'models/frozen_inference_graph_1.pb'
label_path = 'models/classes.pbtxt'
#######################################################################################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["image"])))

savecrop = open('crop_neeeeew.xml', 'w')
writeStart = "<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n<name>imglab dataset</name>\n<comment>Created by imglab tool.</comment>\n<images>\n"
savecrop.write(writeStart)

# initialize the model
model = tf.Graph()

with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()

	# load the graph from disk
	with tf.gfile.GFile(model_path, "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(label_path)
num_classes = str(labelMap).count('item')
categories = label_map_util.convert_label_map_to_categories(
	labelMap, max_num_classes=num_classes,
	use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a session to perform inference
with model.as_default():
	sess = tf.Session(graph=model)

# grab a reference to the input image tensor and the boxes
# tensor
imageTensor = model.get_tensor_by_name("image_tensor:0")
boxesTensor = model.get_tensor_by_name("detection_boxes:0")

# for each bounding box we would like to know the score
# (i.e., probability) and class label
scoresTensor = model.get_tensor_by_name("detection_scores:0")
classesTensor = model.get_tensor_by_name("detection_classes:0")
numDetections = model.get_tensor_by_name("num_detections:0")

for imagePath in imagePaths:
	
	print(imagePath)
	forder = imagePath.split(os.path.sep)[-2]
	fileName = imagePath.split(os.path.sep)[-1]
	error = False

	writeFilename = "  <image file='{}'>\n".format(imagePath)
	savecrop.write(writeFilename)

	# load the image from disk
	image = cv2.imread(imagePath)
	output = image.copy()
	#output = imutils.resize(output, width=640)
	
	startTime = time.time()
	# prepare the image for detection
	#image = image[0:480, 80:560]
	(H, W) = image.shape[:2]
	image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	image = np.expand_dims(image, axis=0)

	# perform inference and compute the bounding boxes,
	# probabilities, and class labels
	(boxes, scores, labels, N) = sess.run(
		[boxesTensor, scoresTensor, classesTensor, numDetections],
		feed_dict={imageTensor: image})

	# squeeze the lists into a single dimension
	boxes = np.squeeze(boxes)
	scores = np.squeeze(scores)
	labels = np.squeeze(labels)

	# C = 0
	# loop over the bounding box predictions
	for (box, score, label) in zip(boxes, scores, labels):
		# if the predicted probability is less than the minimum
		# confidence, ignore it
		if score < min_confidence:
			continue

		# scale the bounding box from the range [0, 1] to [W, H]
		(startY, startX, endY, endX) = box
		startX = int(startX * W)
		startY = int(startY * H)
		endX = int(endX * W)
		endY = int(endY * H)

		# draw the prediction on the output image
		label = categoryIdx[label]
		idx = int(label["id"])
		showlabel = "{}: {:.2f}".format(label["name"], score)
		print(showlabel)

		cv2.rectangle(output, (startX, startY), (endX, endY), (255, 255, 255), 10)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(output, showlabel, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)

		writeData = "    <box top='{}' left='{}' width='{}' height='{}'>\n      <label>{}</label>\n    </box>\n".format(startY, startX, endX-startX, endY-startY, label["name"])
		savecrop.write(writeData)

	writeFileEnd = "  </image>\n"
	savecrop.write(writeFileEnd)
	
	# show the output image
	output = imutils.resize(output, width=1024)
	#cv2.imwrite('test/{}'.format(fileName), output)
	
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
	# elif key == ord("s"):
	# 	shutil.copy(imagePath, 'img/{}'.format(fileName))

writeEnd = "</images>\n</dataset>"
savecrop.write(writeEnd)
savecrop.close()
