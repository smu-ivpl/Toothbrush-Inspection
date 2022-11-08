## checklist

## file hwakjangja..  .jpg or .png or .bmp
## test_dir name!! check test_dir in visualize.py too!

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd
import cv2
import re
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.applications.efficientnet import EfficientNetB3, EfficientNetB0
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
from PIL import Image, ImageFile
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import brush_visualize
import time

DEFAULT_IMAGE_DIR = "/home/vi/VisionData/NOAH/GONGIN/data_test"
############################################################
#  Configurations
############################################################
class ToothBrushHeadConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "toothbrush_head"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

detect_time = []
def detect_and_color_splash(model, image_path=None, img_file_name=None):
    assert image_path

    print("Running on {}".format(image_path))
    # Read image
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = skimage.io.imread(image_path)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    # Detect objects
    # Run model detection and generate the color splash effect
    start_inference = time.time()
    r = model.detect([image], verbose=1)[0]

    ## check time for inference
    end_inference = time.time()
    dd_time = end_inference - start_inference
    dd_time = np.floor(dd_time*10) / 10
    if "dummy" not in img_file_name:
        print("inference_time :", dd_time," sec for", img_file_name)
        detect_time.append(dd_time)

    # bounding box visualize
    class_names = ['background', 'defect']
    bbox = utils.extract_bboxes(r['masks'])
    file_name_bb = "bb_splash_{}".format(img_file_name)
    save_path_bb = os.path.join(DEFAULT_IMAGE_DIR, 'result', file_name_bb)

    # print("image_path", image_path)

    ######## 주석 바꾸기 ! ###############
    # bv = brush_visualize.display_instances(save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
    brush_visualize.display_instances(save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])

    
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{}".format(img_file_name)
    save_path = os.path.join(DEFAULT_IMAGE_DIR , 'result', file_name)
    skimage.io.imsave(save_path, splash)
    
    
    print("Saved to ", save_path)
    # 주석 씌우기
    # return bv



############################################################
#  sort filenames by human order
############################################################

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]




############################################################
#  classification
############################################################

class_time = []

### 주석 바꾸기!  
def binary_classification(imgname, model): 
# def binary_classification(imgname, model, points):
    test_dir = os.path.join(DEFAULT_IMAGE_DIR+'/cropped/'+imgname)
    cnt_cropped = os.path.join(test_dir+'/test')
    

    if not os.path.exists(cnt_cropped):
        os.makedirs(cnt_cropped)

    cropped = len(os.listdir(cnt_cropped))

    path =  "/home/vi/VisionData/NOAH/GONGIN/data_test/result/"
    res_path = os.path.join(path+'bb_splash_'+imgname+'.bmp')
    
    or_path = "/home/vi/VisionData/NOAH/GONGIN/data_test/"
    ori_path = os.path.join(or_path + imgname+'.bmp')
        
    res_img = cv2.imread(ori_path, 3)
    input_img = res_img.copy()
    bb_img = cv2.imread(res_path, 3)      
        

    if cropped == 25 or cropped == 34:
        test_datagen = ImageDataGenerator(
            rescale=1 / 255
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(32, 32),
            batch_size=1,
            shuffle=False,
            class_mode=None
        )
        
        #print("check filenames : ", test_generator.filenames)
        start_classification = time.time()
        preds = model.predict_generator(test_generator, steps=len(test_generator.filenames))

        end_classification = time.time()

        ## check time
        inference_time = end_classification - start_classification
        inference_time = np.floor(inference_time *10) /10
        if "dummy" not in imgname:
            print(inference_time," sec for inferencing toothbrush hair")
            class_time.append(inference_time)

        
        test_generator.filenames.sort(key=natural_keys)
        #print("after sort : ", test_generator.filenames)
        # print(preds)
        image_ids = [name.split('/')[-1] for name in test_generator.filenames]
        predictions = preds.flatten()

        error = []
        cnt = []
        for i in range(len(test_generator.filenames)):    
            #print("error? : ", predictions[i])
            if predictions[i] > 0.5:
                error.append('error')
                cnt.append(i)
            else:
                error.append('normal')
        data = {'filename': image_ids, 'true_label': test_generator.classes, 'category': error}

        submission = pd.DataFrame(data)
        #text and bbox on result image 
        
        ################ 주석 씌우기 시작 2 !! #############################
        '''
        for i in cnt:
            x1,x2,y1,y2 = points[i]
            print(f"(x1, y1) = ({x1}, {y1}), (x2, y2) = ({x2}, {y2})")
            cv2.rectangle(res_img, (x1,y1), (x2,y2), (0,0,255), 2)
            

               
        res_img = cv2.resize(res_img, (int(res_img.shape[1]*0.5), int(res_img.shape[0]*0.5)))
        bb_img = cv2.resize(bb_img, (int(bb_img.shape[1]*0.5), int(bb_img.shape[0]*0.5)))
        input_img = cv2.resize(input_img, (int(input_img.shape[1]*0.5), int(input_img.shape[0]*0.5)))
        res_img = cv2.putText(res_img, f"{len(cnt)} error brush detected", (200,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
        
        cv2.imwrite(f"/home/yjkim/NOAH/gongin/datasets/brush/result/bbox_{imgname}.png", res_img)
        cv2.imwrite(f"/home/yjkim/NOAH/gongin/datasets/brush/result/bb_text_{imgname}.png", bb_img)
        
        print(f"brush number {cnt} is error") 
        
        cv2.imshow("input image", input_img)
        cv2.imshow("detected image", bb_img)
        cv2.imshow("result image", res_img)  
          
            
        while(True):
            if cv2.waitKey(1)&0xFF == ord('x'):
                cv2.destroyAllWindows()
                break
        '''
        ################ 주석 해제 끝 2 !! #############################
        
        # final classification! whether error or not
        if (submission['category'] == 'error').any():
            print(imgname,"is error tooth brush")
            return 0

        else:
            return 1
        

        
    else:
        inference_time = 0
        class_time.append(inference_time)
        print(round(inference_time,2), " sec for inferencing toothbrush hair")

        # text on result image 
        # 식모 갯수보다 더 잡았을때 => 식모가 너무 불규칙해서 에러로 잡는 경우
        ################ 주석 씌우기 시작 3!! #############################
        '''
        res_img = cv2.putText(res_img, f"more brush detected", (200,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
        res_img = cv2.resize(res_img, (int(res_img.shape[1]*0.5), int(res_img.shape[0]*0.5)))
        input_img = cv2.resize(input_img, (int(input_img.shape[1]*0.5), int(input_img.shape[0]*0.5)))
        
        cv2.imshow("input image", input_img)
        cv2.imshow("result image", res_img)
        while (True):
            if cv2.waitKey(1) & 0xFF == ord('x'):
                cv2.destroyAllWindows()
                break
         '''
        
        ################ 주석 씌우기 끝 3!! #############################
        return 0
        
        
############################################################
#  main
############################################################

#if __name__ == '__main__':

def head_brush(brush_model, eff_model, img_dir):
    result_path = DEFAULT_IMAGE_DIR+'/result'
    cropped_path = DEFAULT_IMAGE_DIR+'/cropped'

    if not os.path.isdir(result_path):
        os.mkdir(DEFAULT_IMAGE_DIR+'/result')
        os.mkdir(DEFAULT_IMAGE_DIR+'/cropped')


    # each image in folder
    imgname = img_dir.split("/")[-1]
    onlyname = imgname.split(".")[0]
    
    result = []
    submission = {}
    err_toothbrush_total = []
    each_toothbrush_info_total =[]
    ## arrange results and make csv file
    ########## 주석 바꾸기 ! ############
    # bv = detect_and_color_splash(mrcnn_model, image_path=imgname, img_file_name=imgname_png)
    detect_and_color_splash(brush_model, image_path=img_dir, img_file_name=imgname)
    
    ########## 주석 바꾸기 ! ############
    result = binary_classification(onlyname, eff_model)
    # result = binary_classification(onlyname, eff_model, bv)
    
    submission[onlyname] = result
    print("submission : ", submission)


        
    ### compute time
   
