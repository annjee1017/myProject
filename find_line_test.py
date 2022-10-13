# 이미지의 minX 와 maxX 구하기 (find_line_test.py)

from imutils import paths
import numpy as np
import imutils
import cv2, os

#-----------------------------------------------------------------
input_dir = 'img' 
output_dir = 'result_line' 
os.makedirs(output_dir, exist_ok=True)
#-----------------------------------------------------------------

# 이미지 읽어서 그레이스케일 변환
imagePaths = sorted(list(paths.list_images(input_dir)))
num = 0
for idx, imagePath in enumerate(imagePaths):
    print(imagePath)
    num += 1
    
    image = cv2.imread(imagePath)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 자르기
    # imgray = imgray[2800:, :]
    h, w = imgray.shape[:2]
    canny = cv2.Canny(imgray, 65, 255)

    # 허프 선 검출, 직선으로 판단할 최소한의 점은 80
    lines = cv2.HoughLines(canny, 1, np.pi/180, 20)
    
    x_list, y_list = [], []
    try: 
        for line in lines: 
            # 거리와 각도
            r,theta = line[0] 
            # x, y축에 대한 삼각비
            tx, ty = np.cos(theta), np.sin(theta) 
            #x, y 기준(절편) 좌표
            x0, y0 = tx*r, ty*r  

            # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
            x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
            x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)     
            
            #x 좌표 담기
            if x1 > 0 :
                x_list.append(x1)
                min_x, max_x = min(x_list), max(x_list)

            # 선그리기
            # cv2.line(draw_image, (x1, y1), (x1, y2), (255,0,0), 5)

    except:
        min_x, max_x = 0, w   

    # 선그리기
    cv2.line(image, (min_x, 0), (min_x, h), (0,255,0), 5)
    cv2.line(image, (max_x, 0), (max_x, h), (0,255,0), 5)
    text = f'{num}. width (pixels): {max_x-min_x}'
    cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
    
    # 이미지 저장
    cv2.imwrite(f'{output_dir}/output_img_{num}.jpg', image)

    # 결과 출력
    image = imutils.resize(image, width=1024)
    cv2.imshow('image', image)
    print('width: ', max_x-min_x)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
