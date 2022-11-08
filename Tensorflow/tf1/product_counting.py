# USAGE
# python product_counting.py --image 01_image

# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import cv2
import random
import imutils
from imutils import paths
import os
import shutil
import time
import argparse
from PIL import Image
# Object detection imports
from api import object_counting_api
from utils import backbone
from utils import visualization_utils as vis_util

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# set hyperparameters -----------------------------------------------
model_path = 'models/frozen_inference_graph.pb'
label_path = 'models/classes.pbtxt'

num_classes = 1

is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
# roi = 780 #650 # roi line position
roi = 560 # roi line position
deviation = 100 # the constant that represents the object counting area
custom_object_name = 'product' # set it to your custom object name
targeted_objects ='product'

# set model --------------------------------------------------------
detection_graph = tf.Graph()
with detection_graph.as_default():
    graphDef = tf.GraphDef()
    # load the graph from disk
    with tf.gfile.GFile(model_path, "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(label_path)
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=num_classes,
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# create a session to perform inference ------------------------------------
total_passed_objects = 0
object_cnt = 0
first_xmin = 0
with detection_graph.as_default():
    # sess = tf.Session(graph=detection_graph)
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

        detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = detection_graph.get_tensor_by_name("num_detections:0")

        # run object_counting_api ----------------------------------------
        imagePaths = sorted(list(paths.list_images(args["image"])))
        for frame, imagePath in enumerate(imagePaths):
            folder_name = imagePath.split(os.path.sep)[-2]
            file_name = imagePath.split(os.path.sep)[-1]
            # print(f"{frame}: {file_name}")

            # load the image from disk
            image = cv2.imread(imagePath)
            output = image.copy()

            # prepare the image for detection
            (height, width) = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image, axis=0)
            # print(image.shape, '*'*30) # (1, 1024, 1224, 3)

            # perform inference and compute the bounding boxes, probabilities, and class labels
            (boxes, scores, labels, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.        
            counter, csv_line, counting_result, second_xmin = vis_util.visualize_boxes_and_labels_on_image_array_y_axis( frame,
                                                                                                            output,
                                                                                                            is_color_recognition_enabled,
                                                                                                            np.squeeze(boxes),
                                                                                                            np.squeeze(labels).astype(np.int32),
                                                                                                            np.squeeze(scores),
                                                                                                            category_index,
                                                                                                            targeted_objects = targeted_objects,
                                                                                                            y_reference = roi,
                                                                                                            deviation = deviation,
                                                                                                            use_normalized_coordinates=True,
                                                                                                            line_thickness=4)

            # when the object passed over line and counted, make the color of ROI line green
            if counter == 1:                  
                cv2.line(output, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                object_cnt += 1
                if object_cnt == 1:
                    total_passed_objects = total_passed_objects + counter
                    print(f'counting_result: {total_passed_objects} - {counting_result} / {object_cnt} / {abs(second_xmin-first_xmin):.3f}')                    
                    
                #     if abs(second_xmin-first_xmin) > 0.15: #0.280:
                #         total_passed_objects = total_passed_objects + counter
                #         print(f'counting_result: {total_passed_objects} - {counting_result} / {object_cnt} / {abs(second_xmin-first_xmin):.3f}')
                #         first_xmin = second_xmin
                #     else:
                #         print(f'*******************check: {total_passed_objects} - {counting_result} / {counter} / {abs(second_xmin-first_xmin):.3f}')
                #         first_xmin = 0
            else:
                cv2.line(output, (0, roi), (width, roi), (0, 0, 0xFF), 5)
                first_xmin = 0
                object_cnt = 0
            
            # total_passed_objects = total_passed_objects + counter

            # insert information text to video frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                output,
                'Detected ' + custom_object_name + ': ' + str(total_passed_objects),
                (10, 50),
                font,
                0.8,
                (0, 0xFF, 0xFF),
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
                )               
            
            cv2.putText(
                output,
                'ROI Line',
                (545, roi-10),
                font,
                0.6,
                (0, 0, 0xFF),
                2,
                cv2.LINE_AA,
                )

            cv2.imshow('object counting', output)
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{file_name}', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
