import sys, os
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import tensorflow.keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math
import time
import warnings
from multiprocessing import Process
warnings.filterwarnings('ignore')
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import crack_visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = "./logs/crack"

DEFAULT_IMAGE_DIR = "/home/vi/VisionData/image/CAM1"
############################################################
#  Configurations
############################################################


class ToothBrushCrackConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "toothbrush_crack"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1+ 4 #1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


class Toothbrushcrack(Process):

    def __init__(self):
        Process.__init__(self)


    def color_splash(self, image, mask):
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


    def detect_and_color_splash(self, model, image_path, img_file_name, img_dir):
        import cv2
        # Run model detection and generate the color splash effect
        print("Running on {}".format(img_dir))
        # Read image
        if os.path.exists(img_dir):
            image = skimage.io.imread(img_dir)
        else :
            print("already finished in t1.py")
            return 0
        # Detect objects
        #start = time.time()
        r = model.detect([image], verbose=1)[0]
        #end = time.time()
        #total_time = end - start

        #times = [total_time]
        #sumTime = 0
        #for t in times:
        #    sumTime+=t

        #meanTime = sumTime / len(times)
        #print(f"{total_time:.5f} sec")
        #print("mean time: ", meanTime)
        # bounding box visualize
        class_names = ['bg','1','2','3','4']
        bbox = utils.extract_bboxes(r['masks'])
        file_name_bb = "bb_splash_{}".format(img_file_name)
        save_path_bb = os.path.join(image_path, 'result', file_name_bb)

        display_img = cv2.imread(img_dir, 3)
        display_img = cv2.resize(display_img, (int(display_img.shape[1]*0.4), int(display_img.shape[0]*0.4)))

        # cv2.imshow("input image", display_img)

        ## for check
        #print("class_ids", r['class_ids'])
        lbList=[]
        lbList=crack_visualize.display_instances(save_path_bb, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        print("label list: ", lbList)
        # skimage.io.imsave(save_path_bb, bb_splash)
        # Color splash
        splash = self.color_splash(image, r['masks'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = "splash_{}".format(img_file_name)
        save_path = os.path.join(image_path, 'result', file_name)
        #skimage.io.imsave(save_path, splash)

        # classification to error list & normality list
        for class_n in lbList:
            if(class_n=='1' or class_n=='2' or class_n=='3'):
                print("Saved to ", save_path)
                return 1

        #print("Saved to ", save_path)
        
        print(" ############################# t2.py finished #############################")
        return 0



############################################################
#  Testing
############################################################

#if __name__ == '__main__'
def head_crack(**kwargs):

    headcrack_model = kwargs['model_hcrack'] 
    folder_dir = kwargs['default_image_dir'] 
    in_que1 = kwargs['que_in_1']
    out_que1 = kwargs['que_out_1']
    in_que = kwargs['que_in_2']
    out_que = kwargs['que_out_2']
    in_que3 = kwargs['que_in_3'] 
    out_que3 = kwargs['que_out_3'] 
    in_que4 = kwargs['que_in_4'] 
    out_que4 = kwargs['que_out_4']

    CAM1 = kwargs['cam1']
    CAM2 = kwargs['cam2']
    CAM3 = kwargs['cam3']

    while not kwargs['stop_event'].wait(1e-9):
        if in_que.qsize() > 0:
            print(" ############################# t2.py start! #############################")                    
            img_dir = in_que.pop()
            
            if not os.path.exists(img_dir):
                print(img_dir.split("/")[-1], "is not exists!")
                continue

            
            imgname = img_dir.split("/")[-1]
            onlyname = imgname.split(".")[0]
            error= kwargs['cmodel'].detect_and_color_splash(headcrack_model, image_path=folder_dir,  img_file_name=imgname, img_dir = img_dir)
            cam2_num = int(imgname.split("_")[0]) + 5
            cam3_num = int(imgname.split("_")[0]) + 10
            cam2_dir = os.path.join(CAM2+str(cam2_num).zfill(5)+"_&Cam2Img.bmp")
            cam3_dir = os.path.join(CAM3+str(cam3_num).zfill(5)+"_&Cam3Img.bmp")
            if error:
                if not os.path.exists(img_dir):
                    continue
                
                print(f"{imgname} has head crack error")
                if os.path.exists(img_dir):
                    os.rename(img_dir, img_dir.split('.')[0] + '_0100.png')
                 
                    if os.path.exists(cam2_dir):
                        os.rename(cam2_dir, cam2_dir.split('.')[0] + '_0100.png')
                        
                    if os.path.exists(cam3_dir):
                        os.rename(cam3_dir, cam3_dir.split('.')[0] + '_0100.png')

                #if not in_que1.empty():
                #    in_que1.remove(img_dir)
                    #in_que1.pop()
            else:
                out_que.put(img_dir)
            
            

    #return error
