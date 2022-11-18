# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
import os
from imutils.paths import list_images

TEST_IMAGE_DIR = '01_image/221117'
OUTPUT_DIR = 'output_221117'

crop = [[650, 650, 600, 600]]
for num, path in enumerate(sorted(list(list_images(TEST_IMAGE_DIR)))):
    filepath = os.path.split(path)[0]
    filename = path.split(os.sep)[-1]

    image = cv2.imread(path)

    for num, (x, y, w, h) in enumerate(crop):
        crop_img = image[y:y+h, x:x+w]
        clone = crop_img.copy()
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)[1]

        cv2.imshow("bw", bw)
        key = cv2.waitKey(0)

        # find external contours in the image
        cnts = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours again
        for (i, c) in enumerate(cnts):
            area = cv2.contourArea(c)
            if area > 5000:
                print(area)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 255), 3)  
                cv2.line(clone, (x, cY), (x+w, cY), (0, 255, 255), 3)
                cv2.line(clone, (cX, y), (cX, y+h), (0, 255, 255), 3)

                cv2.drawContours(clone, [c], -1, (255, 0, 0), 3)

        cv2.putText(clone, f"2. Width: {int(w)}px > {int(w)/70:.2f}mm", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)       
        cv2.putText(clone, f"3. Height: {int(h)}px > {int(h)/70:.2f}mm", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)       
        
        os.makedirs(f"{OUTPUT_DIR}/{filepath}", exist_ok=True)
        cv2.imwrite(f"{OUTPUT_DIR}/{filepath}/{filename}", clone)
        
        # clone = imutils.resize(clone, width=1024)
        cv2.imshow("Contours", clone)
        key = cv2.waitKey(0)
        if key == 'q':
            break                   
