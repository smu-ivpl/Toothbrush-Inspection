from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Process
import os
import time
MAX_BUF_SIZE = 100
DEFAULT_IMAGE_DIR = '/home/vi/VisionData/image/' 

class ImageQue:
    def __init__(self):
        self.que = deque(maxlen=MAX_BUF_SIZE)

    def put(self, image):
        self.que.append(image)

    def pop(self):
        return self.que.popleft()
            
    def empty(self):
        return False if self.que else True

    def qsize(self):
        return len(self.que)
        
    def qtolist(self):
        return list(self.que)
        
    def remove(self, rm):
        return self.que.remove(rm)


######que
in_que_1 = ImageQue()
out_que_1 = ImageQue()

in_que_2 = ImageQue()
out_que_2 = ImageQue()

in_que_3 = ImageQue()
out_que_3 = ImageQue()

in_que_4 = ImageQue()
out_que_4 = ImageQue()



class StackingQue(Process):
    def __init__(self):
        Process.__init__(self)

    def run(que, item):
        que.put(item)
        #print("size = ", que.qsize())


class Target():
    watchDir = DEFAULT_IMAGE_DIR + 'CAM1/'
    watchDir2 = DEFAULT_IMAGE_DIR + 'CAM2/'
    watchDir3 = DEFAULT_IMAGE_DIR + 'CAM3/'

    def __init__(self):
        self.observer = Observer()
        self.observer2 = Observer()
        self.observer3 = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDir, recursive=True)
        self.observer.start()
        
        event_handler2 = Handler2()
        self.observer2.schedule(event_handler2, self.watchDir2, recursive=True)
        self.observer2.start()

        event_handler3 = Handler3()
        self.observer3.schedule(event_handler3, self.watchDir3, recursive=True)
        self.observer3.start()
        
        
        try:
            while True:
                time.sleep(1)

        except:
            self.observer.stop()
            print("Error")
            self.observer.join()


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        # print(event)
        if event.src_path.endswith('.bmp') and event.is_directory is False and "result" not in str(event.src_path):
            cam1_image_dir = event.src_path

            StackingQue.run(in_que_1, cam1_image_dir)
            StackingQue.run(in_que_2, cam1_image_dir)
            
            # print('out_que_1 size: ', out_que_1.qsize())
            # print('out_que_2 size: ', out_que_2.qsize())
            '''
            for q1 in range(out_que_1.qsize()):
                output1 = out_que_1.pop()
                print('queue output1: ', output1)
            
            for q2 in range(out_que_2.qsize()):
                output2 = out_que_2.pop()
                print('queue output2: ', output2)
            '''


class Handler2(FileSystemEventHandler):
    i = 0
    def on_created(self, event):
        # print(event)
        if event.src_path.endswith('.bmp') and event.is_directory is False:
            Handler2.i += 1
            #print('Handler2.i: ', Handler2.i)
            cam2_image_dir = event.src_path
            if Handler2.i > 5:    
                StackingQue.run(in_que_3, cam2_image_dir)

            # print('out_que_3 size: ', out_que_3.qsize())
            '''
            for q3 in range(out_que_3.qsize()):
                output3 = out_que_3.pop()
                print('queue output3: ', output3)
            '''


class Handler3(FileSystemEventHandler):
    i = 0
    def on_created(self, event):
        # print(event)
        if event.src_path.endswith('.bmp') and event.is_directory is False:
            Handler3.i += 1
            #print('Handler3.i: ', Handler3.i)
            
            cam3_image_dir = event.src_path
            print('cam3_image_dir: ', cam3_image_dir)
            if Handler3.i > 10:    
                StackingQue.run(in_que_4, cam3_image_dir)
                #print('in_que_4 size: ', in_que_4.qsize())
            
            # print('out_que_4 size: ', out_que_4.qsize())
            '''
            for q4 in range(out_que_4.qsize()):
                output = out_que_4.pop()
                print('queue output4: ', output4)
            '''



def Checking():
    
    w = Target()
    w.run()
    

