import traceback
from tensorflow.keras.models import load_model
from imutils.paths import list_images
from imutils import resize
import numpy as np
import time
import cv2
import os

# Hyper-parameters #-----------------------
TEST_IMGS = '01_image_cont_png'
MODEL_DIR = 'xcp_model/train_98.h5'
OUTPUT_DIR = 'xcp_299'

m_cx, m_cy = 1224, 1024
px2mm = 8
#-------------------------------------------

def get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
	try:
		if x12==x11 or x22==x21:        
			print('1 delta x=0')     
			if x12==x11:            
				cx = x12
				m2 = (y22 - y21) / (x22 - x21)            
				cy = m2 * (cx - x21) + y21
				# print('2')
				return cx, cy        		
			if x22==x21:  
				# print('3')          
				cx = x22
				m1 = (y12 - y11) / (x12 - x11)            
				cy = m1 * (cx - x11) + y11
				return cx, cy    
		m1 = (y12 - y11) / (x12 - x11)    
		m2 = (y22 - y21) / (x22 - x21)    
		if m1==m2:        
			print('4 parallel')        
			return None    
		# print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)    
		cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
		cy = m1 * (cx - x11) + y11
		return cx, cy
	except:
		print(traceback.format_exc())

os.makedirs(OUTPUT_DIR, exist_ok=True)

md = load_model(MODEL_DIR, compile=False)
[(_, h, w, c)] = md.layers[0].input_shape 
# print([(_, h, w, c)]) # [(None, 224, 224, 3)]

for test_path in sorted(list(list_images(TEST_IMGS))):
	ins_time = time.time()
	filename = test_path.split(os.path.sep)[-1]

	test_img = cv2.imread(test_path)
	og_image = test_img.copy()
	th, tw, tc = test_img.shape

	test_img = cv2.resize(test_img, (w, h))
	test_img = test_img / 255.0
	test_img = np.expand_dims(test_img, axis=0)
	
	output = md.predict(test_img)
	print(f"1.ins_time: {time.time()-ins_time:.3f}sec")
	# print('\n\n\n', output) #  [[0.34946465 0.82283944 0.8544109  0.8282653  0.5987539  0.3183835  0.79565746 0.536243   0.6041492  0.7381464  0.42160317 0.5084078 ]]

	# put dots
	x1 = int(output[0][4] * tw)
	y1 = int(output[0][5] * th)

	x3 = int(output[0][6] * tw)
	y3 = int(output[0][7] * th)

	x2 = int(output[0][8] * tw)
	y2 = int(output[0][9] * th)

	x4 = int(output[0][10] * tw)
	y4 = int(output[0][11] * th)

	cv2.circle(og_image, (x1, y1), 10, (0, 255, 255), thickness=-1)
	cv2.circle(og_image, (x2, y2), 10, (0, 255, 255), thickness=-1)
	cv2.circle(og_image, (x3, y3), 10, (0, 255, 255), thickness=-1)
	cv2.circle(og_image, (x4, y4), 10, (0, 255, 255), thickness=-1)

	cv2.line(og_image, (x1, y1), (x3, y3), (255,255,0), 5)
	cv2.line(og_image, (x2, y2), (x4, y4), (255,255,0), 5)

	try:
		cx, cy = get_crosspt(x1,y1, x3,y3, x2,y2, x4,y4)
		cx, cy = int(cx), int(cy)
		diff_cx = int(m_cx - cx)
		diff_cy = int(m_cy - cy)
		cv2.circle(og_image, (cx,cy), 20, (0, 255, 0), thickness=-1)

		# master cx, cy
		# cv2.putText(
		# og_image, f"1. master: ({m_cx}, {m_cy})", (80, 100),
		# cv2.FONT_HERSHEY_PLAIN,
		# 3, (255, 255, 255), 5)
		# current cx, cy
		# cv2.putText(
		# 	og_image, f"2. current: ({cx}, {cy})", (80, 150),
		# 	cv2.FONT_HERSHEY_PLAIN,
		# 	3, (255, 255, 255), 5)
		# diff cx,cy to mm
		# cal_txt = f"3. diffence: {diff_cx}px -> {int(diff_cx/px2mm)}mm, {diff_cy}px -> {int(diff_cy/px2mm)}mm"
		# cv2.putText(
		# og_image, cal_txt, (80, 200),
		# cv2.FONT_HERSHEY_PLAIN,
		# 3, (0, 0, 255), 5)

		# print(f"{filename}: {cal_txt}")
		# print(f"ins_time: {time.time()-ins_time:.3f}sec")
		print(f"2. current: ({cx}, {cy})")
		
	except:
		pass

	save_img_name = f'{filename}_{time.time():.5f}'.replace('.', '_') + '.jpg'
	show_img = resize(og_image, width=800)
	cv2.imshow('imshow', show_img)
	cv2.imwrite(os.path.join(OUTPUT_DIR, save_img_name), show_img)
	key = cv2.waitKey(1)

	if key == ord('q'):
		break
