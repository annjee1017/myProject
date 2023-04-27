import argparse
from bs4.element import Tag
from bs4 import BeautifulSoup as bs
import random, cv2, re, os
from bs4 import BeautifulSoup
from copy import deepcopy, copy
from tqdm import tqdm
from numpy import ceil, mean
import imutils

# argparse 모듈을 사용하여 인자를 받아옴
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--xml', required = True) # 인자로 xml 파일을 받음
args = vars(ap.parse_args())

# bs4 모듈을 이용하여 xml 파일을 파싱함
soup_input = bs(open(args['xml']).read(), 'lxml')

writer = None
video_path2 = None
# xml 파일에서 image 태그를 찾아서 반복문 실행
for imageElement in soup_input.findAll('image'):
    image_path = imageElement['file']
    print(image_path)

    # opencv를 사용하여 이미지를 읽어옴
    image = cv2.imread(image_path)

    # label 태그가 있을 경우 label 값을 받아옴, 없으면 빈 문자열을 받아옴
    labelname = imageElement.label
    if labelname == None:
        labelname = ''
    else :
        labelname = labelname.text

    # 이미지 경로에서 영상 경로를 추출함
    video_path = image_path.split('/')[0] # 상위 폴더 이름
    
    # 영상 경로가 다를 때만 영상 1개 생성 후 이어서 저장
    if video_path != video_path2:
        try:
            # 메모리 초기화
            writer.release()
        except AttributeError:
            pass        

        input_img = cv2.imread(image_path)
        H, W, _ = input_img.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        os.makedirs(f'video_output', exist_ok = True)
        output_path = os.path.join(f'video_output', f'{video_path}.mp4')

        writer = cv2.VideoWriter(output_path, fourcc, 15, (W, H), True)

    video_path2 = copy(video_path)

    # 이미지에서 box 태그를 찾아서 반복문 실행
    for box_soup in imageElement.select('box'):
        label = box_soup.label.getText()

        # 라벨이 'NG'일 경우 색깔 지정함
        if 'NG' in label:
            color = (0,255,255)     

        # box 태그에서 위치와 크기 정보를 받아와서 rectangle을 그리고, label을 씀
        x, y, w, h = map(int, (box_soup['left'], box_soup['top'], box_soup['width'], box_soup['height']))
        
        # satrt_x = 30 if x < 50 else x
        # satrt_x = 1910 if satrt_x > 1900 else satrt_x
        # end_x = 1910 if x+w > 1900 else x+w

        if 'OK' in label:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 8)
            cv2.putText(image, 'OK', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 8)
            cv2.putText(image, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)
    
    # 영상 저장
    writer.write(image)

    image = imutils.resize(image, width=1024)
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF	
    if key == ord("q"):
        break

print('\n[INFO] completed')