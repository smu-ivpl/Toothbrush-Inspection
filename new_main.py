import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import threading
import time
from pathlib import Path
from t1 import *
from t2 import *
from t3 import *
from t4 import *
import Loading_models
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DEFAULT_IMAGE_DIR = '/home/vi/VisionData/image/'  # '/home/vi/PycharmProjects/NoahAlgorithm/dataset/'
NUM = 5  # 주기
# checked = []

check_list = []


class Target:
    watchDir = DEFAULT_IMAGE_DIR + 'CAM1/'

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)

        except:
            self.observer.stop()
            print("Error")
            self.observer.join()


class Handler(FileSystemEventHandler):
    i = 0

    def on_moved(self, event):
        print(event)
        # image_dir = os.path.join(DEFAULT_IMAGE_DIR, event.src_path)
        # toothbrush = threading.Thread(target=series(event.src_path))
        # toothbrush.start()

    def on_created(self, event):
        # print(event)
        if event.src_path.endswith('.bmp') and event.is_directory is False and "result" not in str(event.src_path):
            # print("event.src_path",event.src_path)
            # image_dir = os.path.join(DEFAULT_IMAGE_DIR, event.src_path)
            Handler.i += 1
            # check_list.append(event.src_path)
            toothbrush = threading.Thread(target=series(event.src_path, Handler.i))
            toothbrush.start()

    def on_deleted(self, event):
        print(event)

    def on_modified(self, event):
        print(event)

        # if event.src_path.endswith('.bmp') and event.is_directory is False and "result" not in str(event.src_path):
        # print("event.src_path",event.src_path)
        # image_dir = os.path.join(DEFAULT_IMAGE_DIR, event.src_path)
        # Handler.i += 1
        # check_list.append(event.src_path)
        # toothbrush = threading.Thread(target=series(event.src_path, Handler.i))
        # toothbrush.start()


def observers():
    return None


def series(CAM1_dir, i):
    print("start doing series algorithm! : ", i)
    print('This CAM1 imgname is ', CAM1_dir)
    # start = time.time()

    result1 = head_brush(Loading_models.brush_model, Loading_models.eff_model, CAM1_dir)

    if result1 == 1:  # error toothbrush - head

        ## file name overwrite
        os.rename(CAM1_dir, CAM1_dir.split('.')[0] + '_1000.png')
        #os.rename(CAM2_dir, CAM2_dir.split('.')[0] + '_1000.png')
        #os.rename(CAM3_dir, CAM3_dir.split('.')[0] + '_1000.png')


    else:  # error toothbrush - crack

        result2 = head_crack(Loading_models.crack_model, CAM1_dir, DEFAULT_IMAGE_DIR)


        if result2 == 1:
            ## file name overwrite
            os.rename(CAM1_dir, CAM1_dir.split('.')[0] + '_0100.png')
            #os.rename(CAM2_dir, CAM2_dir.split('.')[0] + '_0100.png')
            #os.rename(CAM3_dir, CAM3_dir.split('.')[0] + '_0100.png')

        else: ## side toothbrush

            while True:
                cam2_paths = sorted(Path(DEFAULT_IMAGE_DIR + '/CAM2/').iterdir(), key=os.path.getmtime)
                if len(cam2_paths) > (i + NUM):
                    CAM2_dir = str(cam2_paths[i + NUM - 1])
                    break

            result3 = side_brush(CAM2_dir)

            '''
            while os.path.isfile(CAM2_dir) is True:
                result3 = side_brush(CAM2_dir)
            '''

            if result3 == 1:

                ## file name overwrite
                os.rename(CAM1_dir, CAM1_dir.split('.')[0] + '_0010.png')
                #os.rename(CAM2_dir, CAM2_dir.split('.')[0] + '_0010.png')
                #os.rename(CAM3_dir, CAM3_dir.split('.')[0] + '_0010.png')

            else:

                while True:
                    cam3_paths = sorted(Path(DEFAULT_IMAGE_DIR + '/CAM3/').iterdir(), key=os.path.getmtime)
                    if len(cam3_paths) > (i + 2 * NUM):
                        CAM3_dir = str(cam3_paths[i + 2 * NUM - 1])
                        break

                result4 = back_crack(Loading_models.B_crack_model, CAM3_dir, DEFAULT_IMAGE_DIR)

                '''
                while os.path.isfile(CAM3_dir) is True:
                    result4 = back_crack(Loading_models.B_crack_model, CAM3_dir, DEFAULT_IMAGE_DIR)
                '''

                if result4 == 1:
                    os.rename(CAM1_dir, CAM1_dir.split('.')[0] + '_0001.png')
                    #os.rename(CAM2_dir, CAM2_dir.split('.')[0] + '_0001.png')
                    #os.rename(CAM3_dir, CAM3_dir.split('.')[0] + '_0001.png')
                else:
                    os.rename(CAM1_dir, CAM1_dir.split('.')[0] + '_0000.png')
                    #os.rename(CAM2_dir, CAM2_dir.split('.')[0] + '_0000.png')
                    #os.rename(CAM3_dir, CAM3_dir.split('.')[0] + '_0000.png')
    # end = time.time()
    # print('time is ', end - start)
    print("end doing series algorithm!", i)

if __name__ == '__main__':
    w = Target()
    w.run()

