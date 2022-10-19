#   python ocr_test.py -d 01_image
import cv2
import easyocr
import time
import argparse
from imutils import paths
import numpy as np
import gc
import torch
import os
import random

# del variables
gc.collect()
torch.cuda.empty_cache()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

np.random.seed(42) 
colors = np.random.randint(0, 255, size=(255, 3),dtype="uint8")

reader = easyocr.Reader(['ko'], gpu=True,model_storage_directory='./model')

imagePaths = sorted(list(paths.list_images(args["dataset"])))
cnt = 0
for imagePath in imagePaths:
    cnt += 1

    img = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
    img = img.copy()
    read_img = img[1090:1090+400, 100:100+2200]
    gray_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)

    # results = reader.readtext(gray_img, detail = 0, allowlist ='0123456789.') #detail = 0, coord, score 생략
    results = reader.readtext(gray_img, allowlist ='0123456789.')
    print(results)

    # max_value = max(results, key = len)[0:10]
    # print(max_value)
    
    for res in results : 
        score = res[2]
        if score < 0.5:
            continue

        if 5 <= len(res[1]) <= 10:
            print(f"{cnt}. {res[1]}: , {res[2]:.05f}")

            x = res[0][0][0] 
            y = res[0][0][1] 
            w = res[0][1][0] - res[0][0][0] 
            h = res[0][2][1] - res[0][1][1] 
            
            color_idx = random.randint(100,250) 
            color = [int(c) for c in colors[color_idx]] 
        
            cv2.putText(img, f"Now: {str(res[1])}", (int(x+100), int(y-10+1090)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5) 
            img = cv2.rectangle(img, (x+100, y+1090), (x+100+w, y+1090+h), color, 5)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite(f"output/output_{cnt:05d}.jpg", img)

    # img = cv2.resize(img, dsize=(800,600))
    # cv2.imshow("output",img)

    # key = cv2.waitKey(1) & 0xFF

    # if key == ord('q'):
    #     break 



