import os
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


DEFAULT_IMAGE_DIR = '/home/vi/VisionData/NOAH/GONGIN/data_test/'



class Target:
    watchDir = DEFAULT_IMAGE_DIR
    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDir,
                                                       recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
                image_dir = os.path.join(self.watchDir, imagename)
                toothbrush = threading.Thread(target=series(image_dir))
                toothbrush.start()

        except:
            self.observer.stop()
            print("Error")
            self.observer.join()

class Handler(FileSystemEventHandler):
    def on_moved(self, event):
        print(event)

    def on_created(self, event):
        print(event)

    def on_deleted(self, event):
        print(event)

    def on_modified(self, event):
        print(event)


def observers():
    return None

def series(CAM1_dir, i):
    print("start doing series algorithm! : ", i)
    head_brush(Loading_models.brush_model, Loading_models.eff_model, CAM1_dir)
    head_crack(Loading_models.crack_model, CAM1_dir, DEFAULT_IMAGE_DIR)
    print("end doing series algorithm!", i)

'''
checked = []
i = 0
while(True):
    check = False
    time_list = []
    paths = sorted(Path(DEFAULT_IMAGE_DIR).iterdir(), key=os.path.getmtime)
    
    for file in paths:
      if str(file).endswith('.bmp') and str(file).endswith('.bmp') not in checked:
         time_list.append(str(file))
      else :
          continue

    if len(time_list) == 0:
        continue
    
    print("timelist", time_list)
    
    for images in time_list:
      if "Frame" in images and images not in checked:
          i+=1
      #if "CAM1" in paths and time_list not in checked:
          #print("infinite thread")
          image_dir = os.path.join(DEFAULT_IMAGE_DIR, images)
          toothbrush = threading.Thread(target=series(image_dir, i))
          toothbrush.start()
          checked.append(images)
          check = True
          print("!!###########done!")
          
      elif "Frame" not in images and images not in checked:
          print("this image is not from CAM1")
          checked.append(images)
          
    if check is True:
        print("check check!===================")
        continue


    
    #if len(time_list) == len(checked):
    #  time.sleep(10)
    #  break
    #else:
    #  continue
'''
if __name__ == '__main__':
    w = Target()
    w.run()

