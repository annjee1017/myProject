# -*- coding: utf-8 -*-

# import the necessary packages
import os
from imutils import paths
import random
import cv2
import numpy as np
import shutil

# parameters #-----------------------------------------------------------------------
rootDir = 'dataset' # 데이터 갯수를 맞출 폴더

data_times = 2 # 현재 이미지 수 x data_times
limit_file_len = 4963 # 파일 내 limit_file_len 보다 많은 이미지가 있을 경우, 카피 미실행

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
random.seed(11)
drts_path = os.listdir(rootDir)
imgs_path = sorted(list(paths.list_images(rootDir)))
for drt in drts_path:
    file_list = os.listdir(f"{rootDir}/{drt}")
    file_len = len(file_list)
    print(f"[INFO] current : {drt} = {file_len}")
    
    # 파일 내 데이터가 limit_file_len 갯수보다 적을 때만 데이터 늘리기 
    if int(file_len) < int(limit_file_len):
        if int(file_len*data_times) >= int(file_len):
            copy_count = 0
            copy_num = int((file_len*data_times)-file_len)
            print(f"[INFO] copy file : {drt} >> {int(file_len*data_times)}, {copy_num}")
            random.shuffle(file_list)
            for file in file_list:
                while copy_num > copy_count:
                    img = cv2.imread(f"{rootDir}/{drt}/{file}")
                    
                    # random_brightness
                    if copy_count%2 == 0:
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
                    cv2.imwrite(f"{rootDir}/{drt}/{str(copy_count)}_{file}", final_dst)
                    copy_count += 1
                    # print(count)
    else:
        sub_count = 0
        sub_num = file_len-limit_file_len+1
        print(f"[INFO] sub file : {drt} >> {limit_file_len}, {sub_num}")
        
        img_paths = sorted(list(paths.list_images('{}/{}'.format(rootDir, drt))))
        random.shuffle(img_paths)

        save_dir = 'save/{}/{}'.format(rootDir, drt)
        os.makedirs(save_dir, exist_ok=True)
        for num, img_path in enumerate(img_paths):
            filename = img_path.split(os.path.sep)[-1]
            # print(filename)
            shutil.move(img_path, '{}/{}'.format(save_dir, filename))

            if num == sub_num:
                break        


