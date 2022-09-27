# -*- coding: utf-8 -*-
import os
from imutils import paths
import random
import cv2
import numpy as np

# parameters #-----------------------------------------------------------------------
rootDir = 'test_folder' # 이미지를 늘려야하는 폴더

data_times = 2 # 현재 이미지 수 x data_times
limit_file_len = 1000 # 파일 내 limit_file_len 보다 많은 이미지가 있을 경우, 카피 미실행

random_crop_value = 3 # 랜덤 픽셀 범위
random_brightness_value = 10 # 랜덤 픽셀 범위
#------------------------------------------------------------------------------------

def random_crop(img, value=random_crop_value, resize=(300, 300)):
    # print(img.shape[0]) # h, w, c
    val_h = random.randint(-value, value)
    val_w = random.randint(-value, value)
    
    dst_h = img.shape[0] + val_h
    dst_w = img.shape[1] + val_w
    
    crop_img = img[:dst_h, :dst_w]
    dst = cv2.resize(crop_img, dsize=resize) # w, h
    return dst

def random_brightness(img, value=random_brightness_value):
    val_bright = random.randint(0, value)
    val_dark = random.randint(0, value)

    array_bright = np.full(img.shape, (val_bright, val_bright, val_bright), dtype = np.uint8 )
    array_dark = np.full(img.shape, (val_dark, val_dark, val_dark), dtype = np.uint8 )

    dst_bright = cv2.add(img, array_bright)
    dst_dark = cv2.subtract(img, array_dark)

    return dst_bright, dst_dark

# run #---------------------------------------------------------------------------------------------
drts_path = os.listdir(rootDir)
imgs_path = sorted(list(paths.list_images(rootDir)))
for drt in drts_path:
    file_list = os.listdir(f"{rootDir}/{drt}")
    file_len = len(file_list)
    print(f"[INFO] current : {drt} = {file_len}")
    
    if int(file_len) < int(limit_file_len):
        if int(file_len*data_times) >= int(file_len):
            copy_num = int((file_len*data_times)-file_len)
            print(f"[INFO] copy file : {drt} >> {int(file_len*data_times)}, {copy_num}")
            count = 0
            for file in file_list:
                if copy_num > count:
                    img = cv2.imread(f"{rootDir}/{drt}/{file}")
                    
                    # random_brightness
                    if count%2 == 0:
                        # bright img
                        dst_bright, _ = random_brightness(img)
                        # cv2.imwrite('dst_bright.jpg', dst_bright)
                    else:
                        # dark img
                        _, dst_bright = random_brightness(img)
                        # cv2.imwrite('dst_dark.jpg', dst_bright)
                    # random_crop
                    img_h, img_w = img.shape[0], img.shape[1]
                    final_dst = random_crop(dst_bright, resize=(img_w, img_h))
                    cv2.imwrite(f"{rootDir}/{drt}/{str(count)}_{file}", final_dst)
                    count += 1
                    # print(count)


