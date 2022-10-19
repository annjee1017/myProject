# python test_segmentation.py

# pip install patchify
# pip install segmentation-models
from ipaddress import v6_int_to_packed
import keras.backend as K
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import time, sys, cv2, os, gc, imutils
from scipy import spatial
from skimage.morphology import  medial_axis
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()


# hyper-parameters------------------------------------------
MODEL_DIR = '04_models/res/epoch_50.h5'
INPUT_DIR = '01_image'
OUTPUT_DIR = '05_output'

standard_width = 20
axis = [2900, 780, 1800, 2000, 512] # x, y, w, h, resize
#-----------------------------------------------------------

def to_skeleton(input_image):
    image = input_image.copy()
    # image = cv2.resize(image, dsize=(resize,resize))

    # to canny
    kernel = np.ones((6,6),np.float32)/25
    dst = cv2.filter2D(image, -1, kernel)
    ret,thresh = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 50, 200)
    # cv2.imwrite('canny.png', canny)

    # to skeleton
    skeleton, distance = medial_axis(thresh, return_distance=True) 
    skeleton = np.array(skeleton*80,  dtype=np.uint8)
    med_dist = distance*skeleton
    # width_dist = (med_dist)*2
    # cv2.imwrite('skeleton.png', skeleton)
    # cv2.imwrite('med_dist.png', med_dist)

    # calculate width
    skeletonWhite = list(zip(*np.nonzero(skeleton)))
    cannyWhite = list(zip(*np.nonzero(canny)))
    alls = [spatial.KDTree(cannyWhite).query(i) for i in skeletonWhite]
    # print(len(alls), '길이')

    width = []
    mask = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    for num, i in enumerate(alls):
        # print(cannyWhite[alls[num][1]])
        width.append(i[0])
        if i[0] <= int(standard_width/2):
            cv2.line(mask, (skeletonWhite[num][1],skeletonWhite[num][0]),  (cannyWhite[alls[num][1]][1], cannyWhite[alls[num][1]][0]), (0, 0, 255), 1)

    return med_dist, mask, width

def find_contours(input_image):
    # find external contours in the image
    output = input_image.copy()
    output = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    object = 0
    for (i, c) in enumerate(cnts):
        # compute the area and the perimeter of the contour
        area = cv2.contourArea(c)
        # draw the contour on the image
        if area > 10:
            cv2.drawContours(output, [c], -1, (255, 0, 0), 2)
            object += 1

    return output, object

def seg_insepction(model_path, input_image):
  K.clear_session()
  gc.collect()

  # load model
  model = load_model(model_path, compile=False) # compile = False -> just for the prediction
  
  # test insepction
  test_img = np.zeros((1, 256, 256, 3), np.uint8)
  pred = model.predict(test_img)

  # inspection
  image = input_image.copy()
  h, w = input_image.shape[:2]
  image = cv2.resize(image, (256, 256))
  image = image.astype('float32') / 255.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = np.expand_dims(image, axis=0)

  pred = model.predict(image)
  pred = pred[0, :, :, 0]
  pred_mask = np.expand_dims(pred, axis=-1)
  pred_mask = np.where(pred_mask > 0.5, 1, 0).astype('uint8')
  # final_output = cv2.bitwise_and(image[0, ...], image[0, ...], mask=pred_mask)  

  predv = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  predv = cv2.resize(predv, (axis[-1], axis[-1])) # 512, 512
  ret, predv = cv2.threshold(predv, 127, 255, cv2.THRESH_BINARY)
  # now_px_check = np.count_nonzero(predv)

  return predv
