from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer

from imutils import paths
import traceback
import os
import imutils
import time

# hyperparametrs #------------------------------------------------------------------------------
input_path = 'dataset/train/good'

config_path = 'results/padim_new.yaml'
weight_path = 'results/weights'
threshold_value = 0.5

save_path = 'results/output'
# ----------------------------------------------------------------------------------------------

class Anomalib():
    def __init__(self) -> None:
        try:
            self.config = config_path        # 파라메터 경로
            self.weight_path = weight_path    # 모델 ckpt 경로
            
        except:
            print(traceback.format_exc())

    def Anomalib_ModelLoad(self) -> None:
        try:
            config = get_configurable_parameters(config_path=self.config)

            self.inferencer: Inferencer

            module = import_module("anomalib.deploy.inferencers.torch")
            TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
            self.inferencer = TorchInferencer(config=config, model_source=f"{self.weight_path}/model.ckpt", meta_data_path=None)#args.meta_data)
            print('\n[INFO] model loading is completed.\n')

        except:
            print(traceback.format_exc())

    def add_label(self, prediction: np.ndarray, scores: float, font: int = cv2.FONT_HERSHEY_PLAIN) -> np.ndarray:
        try:
            text = f"NG score: {scores:.0%}" # / Threshold: {threshold_value}"
            font_size = prediction.shape[1] // 1024 + 1  # Text scale is calculated based on the reference size of 1024
            (width, height), baseline = cv2.getTextSize(text, font, font_size, thickness=font_size // 2)
            label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
            label_patch[:, :] = (255, 255, 255)
            cv2.putText(label_patch, text, (0, baseline // 2 + height), font, font_size, 0, lineType=cv2.LINE_AA)
            prediction[: baseline + height, : baseline + width] = label_patch

            return prediction

        except:
            print(traceback.format_exc())
            return None    

    def infer(self, image_path: Path, inferencer: Inferencer, overlay: bool = True,  threshold=0.5) -> None:
        try:
            # 이미지 검사 (image에 경로를 할당해도 되고, 이미지 np 를 할당해도 됨)
            anomaly_map, score = self.inferencer.predict(image=image_path, superimpose=True, overlay_mask=False)
            output = anomaly_map.copy()

            # print('******************', type(anomaly_map))

            # 어노말리 맵 + 검증값 도출
            output = self.add_label(anomaly_map, score)
            
            # 불량을 빨강으로 보려면 활성화, 불량을 파랑으로 보려면 비활성화
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            return output, score
        except:
            print(traceback.format_exc())
            return None, 0

    def stream(self, image_path):
        # score_list = []
            # print(filename)

        image = cv2.imread(image_path)
        output, score = self.infer(image, self.inferencer, threshold=threshold_value)
        
            # cv2.imwrite('test.jpg', output)
            # score_list.append(score)
            # print(f'{idx:.03d}. {filename}: {score:.2f}')

        # avg_score = f"{sum(score_list)/len(score_list):.2f}"

        return output, score


if __name__ == '__main__':
    anom = Anomalib()

    ckpt_num = len(os.listdir(weight_path))
    # avg_score_list = []
    # for ckpt in range(ckpt_num):
    # print(f'[START] test for epoch = {ckpt}')

    anom.Anomalib_ModelLoad()
    
    image_paths = sorted(list(paths.list_images(input_path)))
    
    for idx, image_path in enumerate(image_paths):
        filename = image_path.split(os.path.sep)[-1]

        stime = time.time()
        output, score = anom.stream(image_path)
        print(f"{filename}: {time.time()-stime:.3f}sec")
        # avg_score_list.append(avg_score)
        
        os.makedirs(f"{save_path}", exist_ok = True)
        cv2.imwrite(f'{save_path}/{filename}', output)

        output = imutils.resize(output, width = 1024)
        cv2.imshow("Anomaly Map", output)
        cv2.waitKey(1)  # wait for any key press

    # print(f'[RESULT] avg score : {avg_score}', '\n', '*'*50)

    # print(f'[BEST CKPT] {max(avg_score_list)}: epoch={avg_score_list.index(max(avg_score_list))}')
    
    # copy the besk ckpt
