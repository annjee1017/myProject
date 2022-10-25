# python predict2.py -n 5 -m 03_model -d 01_image/00_wire_exposure_image

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import warnings
import argparse
import imutils
import random
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
opt = ap.parse_args()

#######################################################################################################
min_confidence = 0.73
model_path = "03_model"
PATH_TO_CFG = f"{model_path}/pipeline.config"
PATH_TO_CKPT = f"{model_path}/checkpoint"
PATH_TO_LABELS = f"{model_path}/classes.pbtxt"
#######################################################################################################

warnings.filterwarnings('ignore')

random.seed(12)

print('Loading model... ', end='')
start_time = time.time()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('\nModel loading done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

print('\nCategory info : ', len(category_index), category_index)
num_classes = len(category_index)

IMAGE_PATHS = sorted(paths.list_images(opt.image))
for num, image_path in enumerate(IMAGE_PATHS):
    # print('Running inference for {}... '.format(image_path), end='')
    print(image_path)

    # load the image from disk
    image = cv2.imread(image_path)
    output = image.copy()
    image_np = np.array(image)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # prepare the image for detection
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    # print(detections.keys())

    boxes = detections['detection_boxes']
    labels = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']


	# scores = detections['']
	# labels = np.squeeze(labels)

    detections['num_detections'] = num_detections
    labels = np.squeeze(detections['detection_classes'].astype(np.int64))
    
    (H, W) = image.shape[:2]

    # loop over the bounding box predictions
    for (box, label, score) in zip(boxes, labels, scores):
		
        # if the predicted probability is less than the minimum confidence, ignore it
        if score < min_confidence:
            continue

        # scale the bounding box from the range [0, 1] to [W, H]
        (startY, startX, endY, endX) = box
        startX = int(startX * W)
        startY = int(startY * H)
        endX = int(endX * W)
        endY = int(endY * H)

        # draw the prediction on the output image
        label += 1
        labelv = category_index[label] 
        idx = int(labelv["id"])

        showlabel = "{} : {:.2f}".format(labelv["name"], score)
        print(f"\nDetected! {showlabel}")

        cv2.rectangle(output, (startX+10, startY+10), (endX-10, endY-10), (0, 0, 255), 10)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(output, showlabel, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)

    # save the output image
    os.makedirs('output', exist_ok = True)
    cv2.imwrite("output/output_{:05d}.jpg".format(num), output)

    # show the output image
    output = imutils.resize(output,width=768)
    cv2.imshow('output', output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # image  = vis_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes']+label_id_offset,
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=200,
    #         min_score_thresh=min_confidence,
    #         agnostic_mode=False)

    
print('\nDone!')

# plt.show()
