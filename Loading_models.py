# -*- coding: utf-8 -*-
import os.path, time
from pathlib import Path
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from t1 import * #ToothBrushHeadConfig
from t2 import * #ToothBrushCrackConfig
from t4 import * #ToothBrushBackCrackConfig
################# LOAD MODELS #######################
DEFAULT_H_BRUSH_MODEL_DIR = '/home/vi/VisionData/NOAH/GONGIN/models/brush/mask_rcnn_toothbrush_head_0020.h5'
DEFAULT_EFF_MODEL_DIR = '/home/vi/VisionData/NOAH/GONGIN/eff3_ep100_2201005_5_acc96.h5'
DEFAULT_H_CRACK_MODEL_DIR = '/home/vi/VisionData/NOAH/GONGIN/models/front_crack/mask_rcnn_toothbrush_crack_0084.h5'
DEFAULT_B_CRACK_MODEL_DIR = '/home/vi/VisionData/NOAH/GONGIN/models/back_crack/mask_rcnn_toothbrush_crack_0069.h5'
DEFAULT_LOGS_DIR = "./logs"


########### Config  ###############

class BInferenceConfig(ToothBrushHeadConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

brush_config = BInferenceConfig()


class CInferenceConfig(ToothBrushCrackConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

crack_config = CInferenceConfig()


class BInferenceConfig(ToothBrushBackCrackConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

back_crack_config = BInferenceConfig()



#### HEAD BRUSH MODEL  ###########
brush_model = modellib.MaskRCNN(mode="inference", config=brush_config,
                              model_dir=DEFAULT_LOGS_DIR)
brush_model.load_weights(DEFAULT_H_BRUSH_MODEL_DIR, by_name=True)

#### HEAD CRACK MODEL  ###########
crack_model = modellib.MaskRCNN(mode="inference", config=crack_config,
                              model_dir=DEFAULT_LOGS_DIR)
crack_model.load_weights(DEFAULT_H_CRACK_MODEL_DIR, by_name=True)

#### BACK CRACK MODEL  ###########
B_crack_model = modellib.MaskRCNN(mode="inference", config=back_crack_config,
                              model_dir=DEFAULT_LOGS_DIR)
B_crack_model.load_weights(DEFAULT_B_CRACK_MODEL_DIR, by_name=True)

##### EFFICIENTNET

efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(32, 32, 3),
    include_top=False,
    pooling='max'
)

eff_model = Sequential()
eff_model.add(efficient_net)
eff_model.add(Dense(units=120, activation='relu'))
eff_model.add(Dense(units=120, activation='relu'))
eff_model.add(Dense(units=1, activation='sigmoid'))
eff_model.summary()

eff_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
eff_model = load_model(DEFAULT_EFF_MODEL_DIR)

############### Done Load #################
