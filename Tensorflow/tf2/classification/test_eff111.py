# python test_tf2.py

import efficientnet.tfkeras
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import math as mt
import imutils
import pickle
import cProfile
import shutil
import time
import cv2
import os

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#-----★★★★★ Hyper-parameters ★★★★★--------------------------------------------------#
INPUT_DIR = 'Future/EASY/TEST' # 하위 폴더 이름이 라벨명과 일치해야함(y_true 값 생성용) eg. MISS/ NG/ OK
OUTPUT_DIR = 'OUTPUT'

MODEL = 'model05/model_resnet50v2/epoch_150.hdf5'
PICKLE = 'model05/model_resnet50v2/model_resnet50v2.pickle'

IMAGE_DIMS = (224, 224)

#-----★★★★★ evaluate code ★★★★★------------------------------------------------------#
# 폴더(라벨) 내 파일 개수 반환
n_test = {}
def get_files_count():
    # each_folder_path = INPUT_DIR
    # file_cnt = len(os.listdir(each_folder_path))
    # return file_cnt

    folders = sorted(os.listdir(INPUT_DIR))
    for folder in folders:        
        each_folder_path = f"{INPUT_DIR}/{folder}"
        file_cnt = len(os.listdir(each_folder_path))
        n_test[folder] = file_cnt
    return folders, n_test   

#-----★★★★★ run code ★★★★★----------------------------------------------------------#
model = load_model(MODEL)
model.summary()

lb    = pickle.loads(open(PICKLE, 'rb').read())
label = lb.classes_

# y_true 값 생성
true_labels, n_test = get_files_count() 
print(n_test) # {'MISS': 336, 'NG': 582, 'OK': 516}

total_y_true = []
# print(true_labels)
for true_label in true_labels:
    y_true_idx = label.tolist().index(true_label) 
    y_true = [y_true_idx]*n_test[true_label]
    total_y_true.extend(y_true) # MISS, NG, OK

# 검증 시작, y_pred 값 생성
total_y_pred = []
imagePaths = sorted(list(paths.list_images(INPUT_DIR)))
for num, imagePath in enumerate(imagePaths): 
    start_time = time.time()
    folderName = imagePath.split(os.path.sep)[-2]
    fileName = imagePath.split(os.path.sep)[-1]

    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]
    output = image.copy()

    crop = [[0, 0, W, H]]
    for key, (x, y, w, h) in enumerate(crop):
        cropimg = image[y:y+h, x:x+w]
        img = cv2.resize(cropimg, (IMAGE_DIMS[0], IMAGE_DIMS[1]), cv2.INTER_LANCZOS4)
        
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # with self.graph_classi.as_default():
        proba = model.predict(img)[0]
        idx_pred = np.argmax(proba) 
        amount = proba[idx_pred]

        total_y_pred.append(idx_pred)
        label_pred = lb.classes_[idx_pred]
        
        showlabel = f'{folderName} > {label_pred} : {amount*100:.2f}%'
        showcolor = (0,255,0) if 'ok' in label_pred.lower() else (0,0,255)
        
        # if 'dummy' in label_pred.lower():
        #     continue
        
        if 'ok' in label_pred.lower():
            cv2.rectangle(output, (x, y), (x+w, y+h), showcolor, 4)
            cv2.putText(output, showlabel, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, showcolor, 1)

        else:
            cv2.rectangle(output, (x, y), (x+w, y+h), showcolor, 4)
            cv2.putText(output, showlabel, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, showcolor, 1)
            
        print(f'{fileName}: {showlabel}, {time.time()-start_time:.3f}sec')

        if folderName != label_pred:
            output_path = f'{OUTPUT_DIR}/{INPUT_DIR}/{folderName}/X_{label_pred}'
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f'{output_path}/{fileName}', output)
        else:
            pass

    # print('-----------------------------------------------')
    output = imutils.resize(output, width = 640)
    cv2.imshow('output', output)
    
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

#-----★★★★★ evaluate code ★★★★★------------------------------------------------------#
total_y_true = np.array(total_y_true)
# print(len(total_y_true), type(total_y_true))
total_y_pred = np.array(total_y_pred)
# print(len(total_y_pred), type(total_y_pred))

print('-----------------------------------------------')
"""
# average option
micro: 전체 클래스에 대한 계산
macro: 각 클래스에 대한 단순평균
weighted: 각 클래스에 속하는 표본의 갯수로 가중평균
accuracy: 정확도. 전체 학습데이터의 개수에서 각 클래스에서 자신의 클래스를 정확하게 맞춘 개수의 비율.
"""

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(total_y_true, total_y_pred) # (정답, 예측값)
print(f'* Accuracy: {accuracy:.3f}%')

# precision tp / (tp + fp) # 높으면 불량 검출율 up
precision = precision_score(total_y_true, total_y_pred, average='macro')
print(f'* Precision: {precision:.3f}%') 

# recall: tp / (tp + fn) # 낮으면 가성 불량율 up
recall = recall_score(total_y_true, total_y_pred, average='macro')
print(f'* Recall: {recall:.3f}%') 

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(total_y_true, total_y_pred, average='macro')
print(f'* F1 score: {f1:.3f}%')

# classification_report
# INPUT_DIR 내 폴더 개수가 라벨 개수와 동일하면: target_names=label
# 다르면 지정: target_names=['class 0', 'class 1']
total_report = classification_report(total_y_true, total_y_pred, target_names=label) 
# total_report = classification_report(total_y_true, total_y_pred, target_names=[label[1], label[2]])
print('\n', total_report)
print('-----------------------------------------------'), 
