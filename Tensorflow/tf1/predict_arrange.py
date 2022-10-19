# import the necessary packages
from PIL import *
import cv2
import numpy as np
import os
from imutils import paths
import imutils
import math
from tqdm import tqdm
from tensorflow.keras.models import load_model
from object_detection.utils import label_map_util
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf

# hyper-parameters------------------------------------------------------
MODEL = 'MODEL1'
INPUT_DIR = f'dataset/{MODEL}'
OUTPUT_DIR = 'output'
dataset_size = (224, 224)

axisC = { 'MODEL1' : [700, 800, 600, 600] }
model_path = f"models/frozen_inference_graph.pb"
label_path = f"models/classes.pbtxt"
min_confidence = 70
# ------------------------------------------------------------------------

def inspection(image, now_model):
    # load model
    model = tf.Graph()
    with model.as_default():
        graphDef = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")

    labelMap = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
        labelMap, max_num_classes= 8,
        use_display_name=True)
    categoryIdx = label_map_util.create_category_index(categories)

    with model.as_default():
        sess = tf.Session(graph=model)
    imageTensor = model.get_tensor_by_name("image_tensor:0")
    boxesTensor = model.get_tensor_by_name("detection_boxes:0")
    scoresTensor = model.get_tensor_by_name("detection_scores:0")
    classesTensor = model.get_tensor_by_name("detection_classes:0")
    numDetections = model.get_tensor_by_name("num_detections:0")

    dst_list = []
    (reY, reX, reH, reW) = axisC[now_model]

    cropimg = image[reY:reY+reH, reX:reX+reW]
    (H, W) = cropimg.shape[:2]
    midX_ori, midY_ori = int(W/2), int(H/2)
    
    save_img = cropimg.copy()

    cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
    cropimg = np.expand_dims(cropimg, axis=0)

    (boxes, scores, labels, N) = sess.run(
    [boxesTensor, scoresTensor, classesTensor, numDetections],
    feed_dict={imageTensor: cropimg})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    labels = np.squeeze(labels)
    
    for (box, score, label) in zip(boxes, scores, labels):
        # score
        score = int(score*100)
        if score < min_confidence:
            continue

        # scale the bounding box from the range [0, 1] to [W, H]
        (startY, startX, endY, endX) = box
        startX = int(startX * W)
        startY = int(startY * H)
        endX = int(endX * W)
        endY = int(endY * H)

        # rotate
        x, y, x2, y2 = startX, startY, endX, endY
        midX = int((x+x2)/2)
        midY = int((y+y2)/2)
        degree = math.degrees(math.atan2(-midY+midY_ori, midX-midX_ori))

        label = categoryIdx[label]
        label = label["name"]
        showlabel = "{}:{:.2f}%".format(label, score)
        print(showlabel)

        if x < 0: x = 0
        if y < 0: y = 0
        if x2 > H: x2 = H
        if y2 > W: y2 = W       
        crop = save_img[y: y2, x: x2]
        crop = imutils.rotate(crop, -degree)
        crop = cv2.resize(crop, dsize=dataset_size, interpolation = cv2.INTER_AREA)
        dst_list.append(crop)

    return dst_list

# run ----------------------------------------------------------------
img_paths = sorted(list(paths.list_images(INPUT_DIR)))
for img_path in tqdm(img_paths):
    filename = img_path.split(os.path.sep)[-1]

    image = cv2.imread(img_path)
    output_list = inspection(image, MODEL)
    for num, output in enumerate(output_list):
        os.makedirs(OUTPUT_DIR, exist_ok = True)
        cv2.imwrite(f'{OUTPUT_DIR}/{num:2d}_{filename}.png', output)

