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

from multiprocessing import Process
#from keras.optimizers import Adam
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
from PIL import Image, ImageFile
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import brush_visualize
import time

DEFAULT_IMAGE_DIR = "/home/vi/VisionData/image/CAM1"
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
#  Class
############################################################
class Toothbrush(Process):
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

    def detect_and_color_splash(self, model, image_path=None, img_file_name=None):
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

        # bounding box visualize
        class_names = ['background', 'defect']
        bbox = utils.extract_bboxes(r['masks'])
        file_name_bb = "bb_splash_{}".format(img_file_name)
        save_path_bb = os.path.join(DEFAULT_IMAGE_DIR, 'result', file_name_bb)

        # print("image_path", image_path)

        ######## 주석 바꾸기 ! ###############
        # bv = brush_visualize.display_instances(save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        crop_list = brush_visualize.display_instances(save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])

        
        splash = self.color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{}".format(img_file_name)
        save_path = os.path.join(DEFAULT_IMAGE_DIR , 'result', file_name)
        #skimage.io.imsave(save_path, splash)
        
        
        #print("Saved to ", save_path)
        # 주석 씌우기
        return crop_list


    ############################################################
    #  classification
    ############################################################

    def binary_classification(self, image_path, imgname, model, crop_list): 
        ## check number of brushes
        try:
            im = Image.open(image_path)
        except :
            print("already finished in t2.py")
            return 0
        
        
        preds = []
        error = []
        if len(crop_list) == 32:
                
            for cc in crop_list:
                #print("cc",cc)
                y1, x1, y2, x2  = cc
                cropImage = im.crop((x1, y1, x2, y2))
                cropImage = cropImage.resize((32,32))
                cropImage = np.array(cropImage)
                cropImage = cropImage[np.newaxis]
                #print("numpy cropimage", cropImage.shape)
                preds.append(model.predict(cropImage))
        
            
            
            for prediction in preds:
                #print("error? : ", prediction.flatten())
                if prediction.flatten() > 0.5:
                    error.append('error')
                else:
                    error.append('normal')
            data = {'filename': imgname,  'category': error}

            submission = pd.DataFrame(data)
            
            # final classification! whether error or not
            if (submission['category'] == 'error').any():
                print(imgname,"is error tooth brush")
                print(" ############################# t1.py finished #############################")
                return 1

            else:
                print(imgname,"is normal tooth brush")
                print(" ############################# t1.py finished #############################")
                return 0
            
        else:
            print(imgname,"is error tooth brush")
            print(" ############################# t1.py finished #############################")
            return 1


def head_brush(**kwargs):
    
    brush_model = kwargs['model_brush']
    eff_model = kwargs['model_eff']
    in_que= kwargs['que_in_1']
    out_que= kwargs['que_out_1']

    in_que2 = kwargs['que_in_2'] 
    out_que2 = kwargs['que_out_2'] 
    in_que3 = kwargs['que_in_3'] 
    out_que3 = kwargs['que_out_3'] 
    in_que4 = kwargs['que_in_4'] 
    out_que4 = kwargs['que_out_4'] 

    CAM1 = kwargs['cam1']
    CAM2 = kwargs['cam2']
    CAM3 = kwargs['cam3']

    result = []
    submission = {}
    ## arrange results and make csv file
    ########## 주석 바꾸기 ! ############
    # bv = detect_and_color_splash(mrcnn_model, image_path=imgname, img_file_name=imgname_png)
    
    
    while not kwargs['stop_event'].wait(1e-9):
        if in_que.qsize() > 0:
            img_dir = in_que.pop()


            if not os.path.exists(img_dir):
                return 0


            result_path = DEFAULT_IMAGE_DIR+'/result'

            if not os.path.isdir(result_path):
                os.mkdir(DEFAULT_IMAGE_DIR+'/result')


            # each image in folder
            imgname = img_dir.split("/")[-1]
            onlyname = imgname.split(".")[0]
            #crop_list = Toothbrush.detect_and_color_splash(brush_model, image_path=img_dir, img_file_name=imgname)
            crop_list = kwargs['bmodel'].detect_and_color_splash(brush_model,image_path=img_dir, img_file_name=imgname)
            result = kwargs['bmodel'].binary_classification(img_dir, onlyname, eff_model, crop_list)
            
            #00055_&Cam3Img.bmp
            cam2_num = int(imgname.split("_")[0]) + 5
            cam3_num = int(imgname.split("_")[0]) + 10
            cam2_dir = os.path.join(CAM2+str(cam2_num).zfill(5)+"_&Cam2Img.bmp")
            cam3_dir = os.path.join(CAM3+str(cam3_num).zfill(5)+"_&Cam3Img.bmp")
            
            if result:
                os.rename(img_dir, img_dir.split('.')[0] + '_1000.png')
                if os.path.exists(cam2_dir):
                    os.rename(cam2_dir, cam2_dir.split('.')[0] + '_1000.png')
                    
                if os.path.exists(cam3_dir):
                    os.rename(cam3_dir, cam3_dir.split('.')[0] + '_1000.png')

                if not in_que2.empty():
                    in_que2.pop()
                    #out_que2()
                out_que.put(img_dir)

            submission[onlyname] = result
            print("submission : ", submission)
            
    #return result

