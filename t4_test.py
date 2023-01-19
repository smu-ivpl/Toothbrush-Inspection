import sys, os
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import tensorflow.keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import cv2
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

############################################################
#  Configurations
############################################################


class ToothBrushBackCrackConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "toothbrush_crack"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # + 4 #1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.98


############################################################
#  Dataset
############################################################

class ToothBrushCrackDataset(utils.Dataset):


    def load_toothbrush_crack(self, dataset_dir, data_list): #, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("toothbrush_crack", 1, "back_crack")
        self.add_class("toothbrush_crack", 2, "normal")

        # Train or validation dataset?
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir)

        ## set absolute json route
        annotations = json.load(open("../models/back_crack/back_region_data_final.json"))

        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # set count i
        i = 0
        # Add images
        for a in annotations:
### check!!
            for name in data_list:
                if a['filename'] == name:

                    if type(a['regions']) is dict:
                        polygons = [r['shape_attributes'] for r in a['regions'].values()]
                        objects = [s['region_attributes']['label'] for s in a['regions'].values()]
                    else:
                        polygons = [r['shape_attributes'] for r in a['regions']]
                        objects = [s['region_attributes']['label'] for s in a['regions']]

                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    # name_dict = {'0': 1, '1': 2}
                    num_ids = [a for a in objects]


                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]
                    # print("image name : ", a['filename'], "ids : ", num_ids)

                    self.add_image(
                        "toothbrush_crack",
                        image_id=a['filename'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids = num_ids)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "toothbrush_crack":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "toothbrush_crack":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']  ## added
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # Get indexes of pixels inside the polygon and set them to 1
            # ---------------------------------------------------------------------------------
            elif p['name'] == 'rect':
                all_points_x = [p['x'], p['x'] + p['width'], p['x'] + p['width'], p['x']]
                all_points_y = [p['y'], p['y'], p['y'] + p['height'], p['y'] + p['height']]
                p['all_points_x'] = all_points_x
                p['all_points_y'] = all_points_y
            # ---------------------------------------------------------------------------------
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        num_ids = np.array(num_ids, dtype=np.int32)

        # print("load_ mask_num_ids", num_ids)
        # return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "toothbrush_crack":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class BackCrack(Process):
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
            '''
            img_pil = Image.fromarray(mask)
            draw = ImageDraw.Draw(img_pil)
            draw.text((60, 70), "check", fill=(255, 0, 0, 0))
            '''

            splash = np.where(mask, image, gray).astype(np.uint8)

        else:
            splash = gray.astype(np.uint8)
        return splash


    def detect_and_color_splash(self, model, image_path=None, img_file_name=None, image_dir=None):

        import cv2
        # Read image
        if not os.path.exists(image_path):
            print(image_path, "not exists!")
            return 0 #, 0
                    
        image = skimage.io.imread(image_path)

        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Detect objects
        start = time.time()
        r = model.detect([image], verbose=1)[0]
        end = time.time()
        total_time = end - start

        times = [total_time]
        sumTime = 0
        for t in times:
            sumTime+=t

        meanTime = sumTime / len(times)

        print(f"{total_time:.5f} sec")
        print("mean time: ", meanTime)
        # bounding box visualize
        class_names = ['bg','1','2']  # ,'3','4']
        bbox = utils.extract_bboxes(r['masks'])
        file_name_bb = "bb_splash_{}".format(img_file_name)
        save_path_bb = os.path.join(image_dir, 'result', file_name_bb)

        #display_img = cv2.imread(image_path, 3)
        #display_img = cv2.resize(display_img, (int(display_img.shape[1]*0.4), int(display_img.shape[0]*0.4)))

        # cv2.imshow("input image", display_img)

        ## for check
        # print("class_ids", r['class_ids'])
        lbList = []
        lbList = crack_visualize.display_instances(save_path_bb, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        print("label list: ", lbList)
        # skimage.io.imsave(save_path_bb, bb_splash)
        # Color splash
        splash = self.color_splash(image, r['masks'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = "splash_{}".format(img_file_name)
        save_path = os.path.join(image_dir, 'result', file_name)
        #skimage.io.imsave(save_path, splash)

        # classification to error list & normality list
        for class_n in lbList:
            if(class_n=='1'):  # or class_n=='2' or class_n=='3'):
                print("Saved to ", save_path)
                return 1 #, meanTime

        print("Saved to ", save_path)
        print("############################## t4.py finished ###################################")    
        return 0 #, meanTime



############################################################
#  Training
############################################################

#if __name__ == '__main__':

realList = []
errList = []
def back_crack(**kwargs):
    #model, img, image_dir
    in_que1= kwargs['que_in_1']
    out_que1= kwargs['que_out_1']
    in_que2 = kwargs['que_in_2'] 
    out_que2 = kwargs['que_out_2'] 
    in_que3 = kwargs['que_in_3'] 
    out_que3 = kwargs['que_out_3'] 
    in_que4 = kwargs['que_in_4'] 
    out_que4 = kwargs['que_out_4'] 
    model = kwargs['model_bcrack']
    image_dir = kwargs['default_image_dir']

    CAM1 = kwargs['cam1']
    CAM2 = kwargs['cam2']
    CAM3 = kwargs['cam3']


    while not kwargs['stop_event'].wait(1e-9):
        print("t4.py inque4 size!!  =  ",in_que4.qsize())
        if in_que4.qsize() > 0:
            print(" ############################# t4.py start! #############################")
            img = in_que4.pop()
            imgname = img.split("/")[-1]
            onlyname, _ = os.path.splitext(imgname)
            imgname_bmp = onlyname + '.bmp'
            # output_imgname = os.path.join(image_path, imgname_png)
            

            if not os.path.exists(img):
                print(imgname, "is not exists!")  
                continue

            
            error = kwargs['bcmodel'].detect_and_color_splash(model, image_path=img,  img_file_name=imgname_bmp, image_dir=image_dir)
            
            
            cam1_num = int(imgname.split("_")[0]) - 10
            cam2_num = int(imgname.split("_")[0]) - 5
            cam1_dir = os.path.join(CAM1+str(cam1_num).zfill(5)+"_&Cam1Img.bmp")
            cam2_dir = os.path.join(CAM2+str(cam2_num).zfill(5)+"_&Cam2Img.bmp")
            if error:
                if not os.path.exists(img):
                    print(f"{imgname_bmp} has back crack! error image")
                    continue
                if os.path.exists(img):
                    os.rename(img, img.split('.')[0] + '_0001.png')
                    if os.path.exists(cam1_dir):
                        os.rename(cam1_dir, cam1_dir.split('.')[0] + '_0001.png')
                    if os.path.exists(cam2_dir):
                        os.rename(cam2_dir, cam2_dir.split('.')[0] + '_0001.png')
              
            else:
                out_que4.put(cam1_dir)
          

