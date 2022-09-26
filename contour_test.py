# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
import os
from imutils.paths import list_images

TEST_IMAGE_DIR = 'test_seg'
OUTPUT_DIR = 'test_output'
os.makedirs(OUTPUT_DIR, exist_ok= True)

total_error = []
for num, path in enumerate(sorted(list(list_images(TEST_IMAGE_DIR)))):
    filename = path.split(os.sep)[-1]
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    # find external contours in the image
    cnts = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    clone = image.copy()

    # loop over the contours again
    cY_1=[]; cY_2=[]; cY_3=[]; cY_4=[]; cX_all=[]
    error_1=[]; error_2=[]; error_3=[]; error_4=[]
    for (i, c) in enumerate(cnts):
        # compute the area and the perimeter of the contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # draw the contour on the image
        if area > 0:
            cv2.drawContours(clone, [c], -1, (255, 0, 0), 3)
            # compute the center of the contour and draw the contour number
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.putText(clone, f"{i+1}", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(clone, f"{cY}", (cX-30, cY-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            # print(f"Contour #{i+1} -- ({cX}, {cY}) / area: {area:.2f} / perimeter: {perimeter:.2f}")
            cX_all.append(cX)        
            if 230 <= cY < 230+150: cY_1.append(cY)
            elif 230+150 <= cY < 230+300: cY_2.append(cY)
            elif 230+300 <=  cY < 230+450: cY_3.append(cY)
            elif 230+450 <= cY < 230+600: cY_4.append(cY)
            else: print('NG', cX, cY)

    # calculate standard Y
    standardY_1 = int((sum(cY_1)-max(cY_1)-min(cY_1))/(len(cY_1)-2))
    standardY_2 = int((sum(cY_2)-max(cY_2)-min(cY_2))/(len(cY_2)-2)) 
    standardY_3 = int((sum(cY_3)-max(cY_3)-min(cY_3))/(len(cY_3)-2))
    standardY_4 = int((sum(cY_4)-max(cY_4)-min(cY_4))/(len(cY_4)-2))

    # draw stadard Y lines
    cv2.line(clone, (min(cX_all), standardY_1), (max(cX_all), standardY_1), (0,255,255), 3)
    cv2.line(clone, (min(cX_all), standardY_2), (max(cX_all), standardY_2), (0,255,255), 3)
    cv2.line(clone, (min(cX_all), standardY_3), (max(cX_all), standardY_3), (0,255,255), 3)
    cv2.line(clone, (min(cX_all), standardY_4), (max(cX_all), standardY_4), (0,255,255), 3)

    # draw the text
    cv2.putText(clone, f"1. Standard Y: {standardY_1}", (min(cX_all)-380, standardY_1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    cv2.putText(clone, f"2. Standard Y: {standardY_2}", (min(cX_all)-380, standardY_2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    cv2.putText(clone, f"3. Standard Y: {standardY_3}", (min(cX_all)-380, standardY_3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    cv2.putText(clone, f"4. Standard Y: {standardY_4}", (min(cX_all)-380, standardY_4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    cv2.putText(clone, f"* Pin: {len(cnts)}ea", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # cv2.putText(clone, f"* StandardY: 1. {standardY_1} / 2. {standardY_2} / 3. {standardY_3} / 4. {standardY_4}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # check error
    for a in cY_1: error_1.append(abs(a-standardY_1)); total_error.append(abs(a-standardY_1))
    for b in cY_2: error_2.append(abs(b-standardY_2)); total_error.append(abs(b-standardY_2))
    for c in cY_3: error_3.append(abs(c-standardY_3)); total_error.append(abs(c-standardY_3))
    for d in cY_4: error_4.append(abs(d-standardY_4)); total_error.append(abs(d-standardY_4))

    # cv2.putText(clone, f"* Max Error: {max(max(error_1), max(error_2), max(error_3), max(error_4))}px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(clone, f"* Max Error: 1. {max(error_1)}px / 2. {max(error_2)}px / 3. {max(error_3)}px / 4. {max(error_4)}px", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(clone, f"* Min Error: 1. {min(error_1)}px / 2. {min(error_2)}px / 3. {min(error_3)}px / 4. {min(error_4)}px", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(clone, f"* Avg Error: 1. {int(sum(error_1)/len(error_1))}px / 2. {int(sum(error_2)/len(error_2))}px / 3. {int(sum(error_3)/len(error_3))}px / 4. {int(sum(error_4)/len(error_4))}px", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # save output images
    cv2.imwrite(f'{OUTPUT_DIR}/output_{filename}', clone)

    print(f"* Avg Error: 1. {int(sum(error_1)/len(error_1))}px / 2. {int(sum(error_2)/len(error_2))}px / 3. {int(sum(error_3)/len(error_3))}px / 4. {int(sum(error_4)/len(error_4))}px")

    # show the output image
    clone = imutils.resize(clone, width=1024)
    cv2.imshow("Contours", clone)
    key = cv2.waitKey(1)
    if key == 'q':
        break

print()
print(f"MIN ERROR: {min(total_error)}")
print(f"MAX ERROR: {max(total_error)}")
