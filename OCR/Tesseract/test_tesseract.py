import cv2
import pytesseract
import argparse
from imutils import paths
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", required=True,	help="path to img")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["img"])))
for (num, imagePath) in enumerate(imagePaths):
	#print(imagePath)
    img = cv2.imread(imagePath)
    (h, w, _) = img.shape

    # 이미지 흑백화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 : 노이즈 줄이기
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # OCR
    chars = pytesseract.image_to_string(blurred, lang='kor', config='--psm 7 --oem 0')
    print(chars)

    # 음절
    boxes = pytesseract.image_to_boxes(blurred, lang='kor')

    for b in boxes.splitlines():
        b = b.split(' ')
        output = cv2.rectangle(blurred, (int(b[1]), h-int(b[2])), (int(b[3]), h-int(b[4])), (255, 0, 0), 2)

    output = cv2.resize(output, (800,600)) #else
    cv2.imshow('img', img)
    cv2.imshow('output', output)

    key = cv2.waitKey(0) & 0xFF            
    if key == ord("q"):
        break
