# USAGE
# python crop.py -i image

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import random
from imutils import paths
import time
import shutil
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="image directory")
args = vars(ap.parse_args())

img_paths = sorted(list(paths.list_images(args['input'])))

x= 600; y= 0; w= 1200; h= 1200
for path in img_paths:
    filename = os.path.basename(path)
    # print(filename)
    filepath = os.path.split(path)[0]
    # print(filepath)
    # basepath = filepath.split(os.path.sep)[1:]
    # # print(basepath)  

    writePath = '02_dataset/'+filepath+'/'
    os.makedirs(writePath, exist_ok=True)

    img = cv2.imread(path)
    crop1 = img[y:y+h, x:x+w] # y h x w 
    cv2.imwrite(f'{writePath}/{filename}', crop1)	

    cv2.imshow('image', crop1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
