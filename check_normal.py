import sys
from threading import Thread, Event
from collections import deque
from pathlib import Path
from multiprocessing import Process, Lock
from t1_test import *
from t2_test import *
from t3_test import *
from t4_test import *
from check_normal import *
import time
import Loading_models
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from mrcnn.deque import ImageQue , Checking
from mrcnn.deque import in_que_1, in_que_2 , in_que_3, in_que_4, out_que_1, out_que_2, out_que_3, out_que_4
MAX_BUF_SIZE = 144

DEFAULT_IMAGE_DIR = '/home/vi/VisionData/image/'
CAM1 = '/home/vi/VisionData/image/CAM1/'
CAM2 = '/home/vi/VisionData/image/CAM2/'
CAM3 = '/home/vi/VisionData/image/CAM3/'

def main():
    args = {}
    
    args['cam1'] = CAM1
    args['cam2'] = CAM2
    args['cam3'] = CAM3
    args['model_brush'] = Loading_models.brush_model
    args['bmodel'] = Toothbrush()
    args['cmodel'] = Toothbrushcrack()
    args['model_eff'] = Loading_models.eff_model
    args['model_hcrack'] = Loading_models.crack_model
    
    args['bcmodel'] = BackCrack()
    args['model_bcrack'] = Loading_models.B_crack_model
    args['smodel'] = SideToothbrush()
    args['default_image_dir'] = DEFAULT_IMAGE_DIR
    args['que_in_1'] = in_que_1
    args['que_out_1'] = out_que_1
    args['que_in_2'] = in_que_2
    args['que_out_2'] = out_que_2
    args['que_in_3'] = in_que_3
    args['que_out_3'] = out_que_3
    args['que_in_4'] = in_que_4
    args['que_out_4'] = out_que_4
    
    args['check_norm'] = CheckingNorm()
    args['stop_event'] = Event()

    que = Thread(target=Checking)
    t1 = Thread(target=head_brush, kwargs=args)
    t2 = Thread(target=head_crack, kwargs=args)
    t3 = Thread(target=side_brush, kwargs=args)
    t4 = Thread(target=back_crack, kwargs=args)
    norm = Thread(target=check_norm, kwargs=args)

    que.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    norm.start()
    

if __name__ == "__main__":
    main()
