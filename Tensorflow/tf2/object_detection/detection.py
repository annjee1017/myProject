import cv2
import numpy as np
from queue import Queue
from PIL import Image, ImageTk

from _thread import *
from threading import Thread

import tensorflow as tf
from object_detection.utils import label_map_util

import keras.backend as K

import configparser
import pickle
import gc
import os
import glob
import imutils
from imutils import paths

import traceback
import logging
import time
from datetime import datetime

def logger_object():
    os.makedirs('./log', exist_ok=True)
    logger = logging.getLogger('log_detection')
    logger.setLevel(logging.INFO)

    nowtime = datetime.now().strftime('%Y_%m_%d')
    file_handler = logging.FileHandler(f'./log/log_{nowtime}.log')
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# setting log
logger = logger_object() 

class ObjectDetection():
	def __init__(self, alias, setvalue_path, tensor_version=1):
		self.myname = alias
		self.setvalue_path = setvalue_path
		
		self.model_comp = False
		self.ok_img = None
		self.ng_img = None
		
		# setting tensorflow version gpu memory
		if tensor_version == 1:
			import tensorflow.compat.v1 as tf
			self.tf = tf
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
		elif tensor_version == 2:
			import tensorflow as tf
			self.tf = tf
			gpus = tf.config.experimental.list_physical_devices('GPU')
			tf.config.experimental.set_memory_growth(gpus[0], True)
			K.clear_session()
		else:
			raise Exception('tensor version must be intvalue 1 or 2')

		# set setval config
		self.setvalue_config = configparser.ConfigParser()
		self.setvalue_config.read(self.setvalue_path, encoding='utf-8')
		self.create_setting_config()


	def read_label_map(self, label_map_path):
		try:
			item_id = None
			item_name = None
			items = {}

			with open(label_map_path, "r") as file:
				for line in file:
					line.replace(" ", "")
					if line == "item {":
						pass
					elif line == "}":
						pass
					elif "id" in line:
						item_id = int(line.split(":", 1)[1].strip())
					elif "name" in line:
						item_name = line.split(" ")[-1].replace("'", " ").strip()
					if item_id is not None and item_name is not None:
						items[item_id] = item_name
						item_id = None
						item_name = None
			return items
		except:
			text = f"read_label_map: {traceback.format_exc()}"
			logger.error(text)						

	def create_setting_config(self, model_name = None):
		try:
			# assign model name
			if model_name != None:
				self.model_name = model_name
			else:
				self.model_name = 'NONE'

			# check if ini file is existing
			if not os.path.exists(self.setvalue_path):
				with open(self.setvalue_path, 'w') as f:
					self.setvalue_config.write(f)

			# import cam_setvalue ini file
			# self.setvalue_config = configparser.ConfigParser()
			# self.setvalue_config.read(self.setvalue_path, encoding='utf-8')
			model_setting = eval(self.setvalue_config[self.model_name]['model_setting'])

			# read trained labels
			label_map_path = str(self.setvalue_config[self.model_name]['label_path'])	
			label_map = self.read_label_map(label_map_path)

			# add model_setting_value, if model_setting_value is None
			if model_setting == {} : # or len(label_map.keys()) :
				model_setting_value = {'total': 60} # default value
				for l in label_map.values():
					model_setting_value.setdefault(l, 60)

				self.setvalue_config.set(self.model_name, 'model_setting', str(model_setting_value))
				with open(self.setvalue_path, 'w') as f:
					self.setvalue_config.write(f)
		except:
			text = f"create_setting_config: {traceback.format_exc()}"
			logger.error(text)		

	def model_load(self, model_path, label_path, model_name = None):
		try:
			# initialize the model
			model = self.tf.Graph()

			# assign model name
			if model_name != None:
				self.model_name = model_name
			else:
				self.model_name = model_path.split('/')[-2]

			with model.as_default():
				try:
					K.clear_session()
					gc.collect()
				except:
					pass

				# initialize the graph definition
				graphDef = self.tf.GraphDef()

				# load the graph from disk
				with tf.io.gfile.GFile(model_path, "rb") as f:
					serializedGraph = f.read()
					graphDef.ParseFromString(serializedGraph)
					self.tf.import_graph_def(graphDef, name="")

				# load the class labels from disk
				labelMap = label_map_util.load_labelmap(label_path)
				self.num_classes = len(labelMap.item)
				self.label_list = [it.name.lower() for it in labelMap.item]
				categories = label_map_util.convert_label_map_to_categories(
					labelMap, max_num_classes=self.num_classes,
					use_display_name=True)
				self.categoryIdx = label_map_util.create_category_index(categories)

				# create a session to perform inference
				with model.as_default():
					self.sess = self.tf.Session(graph=model)

				self.imageTensor = model.get_tensor_by_name("image_tensor:0")
				self.boxesTensor = model.get_tensor_by_name("detection_boxes:0")

				# for each bounding box we would like to know the score
				# (i.e., probability) and class label
				self.scoresTensor = model.get_tensor_by_name("detection_scores:0")
				self.classesTensor = model.get_tensor_by_name("detection_classes:0")
				self.numDetections = model.get_tensor_by_name("num_detections:0")

				# creating config file
				self.create_setting_config()

				# creating result dictionay
				self.rst_dict = {name.lower(): '' for name in self.label_list 
								if 'ok' not in name.lower()}

				self.model_comp = True

				# test inspection
				self.test_inspection(np.zeros((1000, 1000, 3), np.uint8))
				print(f'[INFO] {self.myname} model_load comp')
		except:
			text = f"model_load: {traceback.format_exc()}"
			logger.error(text)
			print(f'[ERROR] {self.myname} model_load is failed')

	def test_inspection(self, img):
		try:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = np.expand_dims(img, axis=0)
			(boxes, scores, labels, N) = self.sess.run(
				[self.boxesTensor, self.scoresTensor,
					self.classesTensor, self.numDetections],
				feed_dict={self.imageTensor: img})

			# squeeze the lists into a single dimension
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			labels = np.squeeze(labels)
		except:
			text = f"test_inspection: {traceback.format_exc()}"
			logger.error(text)

	def inspection(self, img):
		try:
			# read seting config
			self.setvalue_config.read(self.setvalue_path, encoding='utf-8')

			ins_img = img.copy()

			(H, W) = ins_img.shape[:2]

			ins_img = cv2.cvtColor(ins_img, cv2.COLOR_BGR2RGB)
			ins_img = np.expand_dims(ins_img, axis=0)

			(boxes, scores, labels, N) = self.sess.run(
				[self.boxesTensor, self.scoresTensor,
					self.classesTensor, self.numDetections],
				feed_dict={self.imageTensor: ins_img})

			# squeeze the lists into a single dimension
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			labels = np.squeeze(labels)

			showlabel = 'OK'
			for (box, score, label) in zip(boxes, scores, labels):
				label = self.categoryIdx[label]
				idx = int(label["id"])
				label = label['name']

				setting_value = eval(self.setvalue_config[self.model_name]['model_setting'])

				total_conf = setting_value['total']
				if score < total_conf/100:
					continue

				if 'ok' in label.lower(): 
					continue

				# for each_label, each_conf in setting_value.items():
				# 	# print(each_label, each_conf)
				# 	if label == each_label and score < each_conf/100 :
				# 		print('    1. dectection lib:', label, score)					
				# 		continue

				(startY, startX, endY, endX) = box
				startX = int(startX * W)
				startY = int(startY * H)
				endX = int(endX * W)
				endY = int(endY * H)

				showlabel = f"{label} = {score*100:.2f}%"
				print('    2. dectection lib: ', showlabel)
				color = (0,0,255)
				cv2.rectangle(img, (startX, startY), (endX, endY), color, 4)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.putText(img, 'NG', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
				# cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255,255,255), 10)
				showlabel = 'NG'

			return img, showlabel
			
		except:
			text = f"inspection: {traceback.format_exc()}"
			logger.error(text)

if __name__ == '__main__':
	INPUT_DIR = 'test_data' 
	MODEL_NAME = 'NONE'
	MODEL_CONF_PATH = 'config/cam1_odc_config.INI'

	model_config = configparser.ConfigParser()
	model_config.read(MODEL_CONF_PATH, encoding='utf-8')

	model_path = eval(model_config[MODEL_NAME]['model_path'])
	label_path = eval(model_config[MODEL_NAME]['label_path'])
	
	# 1. tensorflow 1
	ODC = ObjectDetection('cam1', MODEL_CONF_PATH, 1)
	ODC.model_load(model_path, label_path, MODEL_NAME)

	# 2. inspect image
	imagePaths = sorted(list(paths.list_images(INPUT_DIR)))
	for num, img_path in enumerate(imagePaths):
		# print(img_path)
		
		image = cv2.imread(img_path)
		output, result = ODC.inspection(image)


		output = imutils.resize(output, width = 800)
		cv2.imshow('show', output)
		key = cv2.waitKey(0)

		if key == ord('q'):
			exit()
