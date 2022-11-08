# -*- coding: utf-8 -*-
import os.path, time
import Loading_models
from pathlib import Path
import threading
from t1 import *
from t2 import *
from t3 import *
from t4 import *

print("####################Process Start!#########################")
#print(Loading_models.brush_model)

# 이름이 CAM1 이면서 1초다음에 생성된..!
DEFAULT_IMAGE_DIR = '/home/vi/VisionData/NOAH/GONGIN/data_test/'
image_path = DEFAULT_IMAGE_DIR

def series():
    CAM1_dir = "/home/vi/VisionData/NOAH/GONGIN/data_test/Frame_0324_20220415145634.bmp"
    CAM2_dir = "/home/vi/VisionData/NOAH/GONGIN/data_test/Frame_0008_20220218103332.bmp"
    CAM3_dir = "/home/vi/VisionData/NOAH/GONGIN/data_test/Frame_0324_20220415145634.bmp"
  
    head_brush(Loading_models.brush_model, Loading_models.eff_model, CAM1_dir)
    head_crack(Loading_models.crack_model, CAM1_dir, DEFAULT_IMAGE_DIR)
    side_brush(CAM2_dir)
    back_crack(Loading_models.B_crack_model, CAM3_dir, DEFAULT_IMAGE_DIR)





checked = []
while(True):
    time_list = []
    paths = sorted(Path(DEFAULT_IMAGE_DIR).iterdir(), key=os.path.getmtime)
    
    for file in paths:
      if str(file).endswith('.bmp'):
        time_list.append(str(file))
    
    print("timelist", time_list)
    
    for images in time_list:
      if "Frame" in images and images not in checked:
      #if "CAM1" in paths and time_list not in checked:
          print("????????????????????????infinite thread")
          toothbrush = threading.Thread(target=series())
          toothbrush.start()
          checked.append(images)
          continue
      elif "Frame" not in images and images not in checked:
          print("this image is not from CAM1")
          checked.append(images)
      elif "Frame" not in images and images in checked:
          continue
      
      
      
    if len(time_list) == len(checked):
      break
      


