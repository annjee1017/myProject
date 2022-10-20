
import cv2
import numpy as np
import pytesseract
import os
from imutils import paths
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="image directory")
args = vars(ap.parse_args())

def imread(filename, dtype=np.uint8):
	try:
		n = np.fromfile(filename, dtype)
		img = cv2.imdecode(n, cv2.IMREAD_COLOR)
		return img
	except Exception as e:
		print(e)
		return None

def imwrite(filename, img, params=None):
	try:
		ext = os.path.splitext(filename)[1]
		result, n = cv2.imencode(ext, img, params)

		if result:
			with open(filename, mode='w+b') as f:
				n.tofile(f)
			return True
		else:
			return False
	except Exception as e:
		print(e)
		return False

img_paths = sorted(list(paths.list_images(args['input'])))
for path in img_paths:
	filename = os.path.basename(path)
	filepath = os.path.split(path)[0]
	print(filename)

	# 이미지 불러오기
	stime = time.time()
	img_ori = imread(path)
	height, width, channel = img_ori.shape


	# 1. OCR 영역 찾기 ---------------------------------------------------
	# 이미지 흑백화
	gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

	structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

	imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
	imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

	imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
	gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

	# 가우시안블러: 노이즈 줄이기
	img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

	# 이미지 이진화: 구별하기 쉽게  (0, 255)
	img_thresh = cv2.adaptiveThreshold(
		img_blurred, 
		maxValue=255.0, 
		adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
		thresholdType=cv2.THRESH_BINARY, 
		blockSize=19, 
		C=9)

	imwrite('01_img_thresh1.jpg', img_thresh)
	# cv2.imshow('img_thresh', img_thresh)
	# key = cv2.waitKey(0) & 0xFF

	# 윤곽선 찾기
	contours = cv2.findContours(
		img_thresh, 
		mode=cv2.RETR_LIST, 
		method=cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	temp_result = np.zeros((height, width, channel), dtype=np.uint8)
	for i in range(len(contours)):
		cv2.drawContours(temp_result, [contours[i]], 0, (255, 255, 255), 2)

	# 윤곽선의 사각형 범위 찾기
	temp_result = np.zeros((height, width, channel), dtype=np.uint8)
	contours_dict = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
		
		# insert to dict
		contours_dict.append({
			'contour': contour,
			'x': x,
			'y': y,
			'w': w,
			'h': h,
			'cx': x + (w / 2),
			'cy': y + (h / 2)
		})

	imwrite('02_temp_result1.jpg', temp_result)
	# cv2.imshow("temp_result", temp_result)
	# cv2.waitKey(0)

	# 사각형 중 OCR 해야할 영역 찾기
	MIN_AREA = 80
	MIN_WIDTH, MIN_HEIGHT = 1, 8
	MIN_RATIO, MAX_RATIO = 0.25, 1.0

	possible_contours = []
	cnt = 0
	for d in contours_dict:
		area = d['w'] * d['h']
		ratio = d['w'] / d['h']
		
		if area > MIN_AREA \
		and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
		and MIN_RATIO < ratio < MAX_RATIO:
			d['idx'] = cnt
			cnt += 1
			possible_contours.append(d)

	temp_result = np.zeros((height, width, channel), dtype=np.uint8)
	for d in possible_contours:
		cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

	imwrite('03_temp_result1.jpg', temp_result)
	# cv2.imshow("temp_result", temp_result)
	# cv2.waitKey(0)

	# 실제 OCR 해야할 영역 찾기
	MAX_DIAG_MULTIPLYER = 5 # 5
	MAX_ANGLE_DIFF = 10.0 # 12.0
	MAX_AREA_DIFF = 0.5 # 0.5
	MAX_WIDTH_DIFF = 0.8
	MAX_HEIGHT_DIFF = 0.2
	MIN_N_MATCHED = 3 # 3

	def find_chars(contour_list):
		matched_result_idx = []
		
		for d1 in contour_list:
			matched_contours_idx = []
			for d2 in contour_list:
				if d1['idx'] == d2['idx']:
					continue

				dx = abs(d1['cx'] - d2['cx'])
				dy = abs(d1['cy'] - d2['cy'])

				diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

				distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
				if dx == 0:
					angle_diff = 90
				else:
					angle_diff = np.degrees(np.arctan(dy / dx))
				area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
				width_diff = abs(d1['w'] - d2['w']) / d1['w']
				height_diff = abs(d1['h'] - d2['h']) / d1['h']

				if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
				and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
				and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
					matched_contours_idx.append(d2['idx'])

			# append this contour
			matched_contours_idx.append(d1['idx'])

			if len(matched_contours_idx) < MIN_N_MATCHED:
				continue

			matched_result_idx.append(matched_contours_idx)

			unmatched_contour_idx = []
			for d4 in contour_list:
				if d4['idx'] not in matched_contours_idx:
					unmatched_contour_idx.append(d4['idx'])

			unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
			
			# recursive
			recursive_contour_list = find_chars(unmatched_contour)
			
			for idx in recursive_contour_list:
				matched_result_idx.append(idx)
			break

		return matched_result_idx
		
	result_idx = find_chars(possible_contours)

	matched_result = []
	for idx_list in result_idx:
		matched_result.append(np.take(possible_contours, idx_list))

	# visualize possible contours
	temp_result = np.zeros((height, width, channel), dtype=np.uint8)

	for r in matched_result:
		for d in r:
			cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

	imwrite('04_temp_result1.jpg', temp_result)
	# cv2.imshow("temp_result", temp_result)
	# cv2.waitKey(0)        

	# 비스듬한 이미지 정방향으로 정렬하기
	PLATE_WIDTH_PADDING = 1.3 # 1.3
	PLATE_HEIGHT_PADDING = 1.5 # 1.5
	MIN_PLATE_RATIO = 1 # 3
	MAX_PLATE_RATIO = 30 

	plate_imgs = []
	plate_infos = []

	for i, matched_chars in enumerate(matched_result):
		sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

		plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
		plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
		
		plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
		
		sum_height = 0
		for d in sorted_chars:
			sum_height += d['h']

		plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
		
		triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
		triangle_hypotenus = np.linalg.norm(
			np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
			np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
		)
		
		angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
		
		rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
		
		img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
		
		img_cropped = cv2.getRectSubPix(
			img_ori, 
			patchSize=(int(plate_width), int(plate_height)), 
			center=(int(plate_cx), int(plate_cy))
		)
		
		if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
			continue
		
		plate_imgs.append(img_cropped)
		plate_infos.append({
			'x': int(plate_cx - plate_width / 2),
			'y': int(plate_cy - plate_height / 2),
			'w': int(plate_width),
			'h': int(plate_height)
		})


	# 2. OCR ---------------------------------------------------
	# 최종 OCR 영역 설정
	longest_idx, longest_text = -1, 0
	plate_chars = []

	for i, plate_img in enumerate(plate_imgs):
		plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
		plate_img = cv2.resize(plate_img, dsize=(360, 60))

		writePath = f"output/{filepath}/"
		os.makedirs(writePath, exist_ok=True)
		imwrite(writePath+filename, plate_img)
