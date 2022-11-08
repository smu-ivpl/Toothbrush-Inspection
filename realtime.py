import os
import shutil
import time

image_path = "/home/vi/VisionData/NOAH/GONGIN/datasets/back_crack/"
output_path = "/home/vi/VisionData/NOAH/GONGIN/data_test/"
dirs = os.listdir(image_path)


images = [file for file in dirs if file.endswith('.bmp')]



for image in images:
    src = os.path.join(image_path, image)
    dst = os.path.join(output_path, image)
    time.sleep(2)
    shutil.copy(src, dst)
