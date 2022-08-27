from PIL import Image
import os
import cv2 as cv

dir_path= 'chars3/chars3' #路径有修改过可能不太正确了 但是训练集已经建立好了 所以不会有任何影响

for item in os.listdir(dir_path):
    item_path = os.path.join(dir_path, item) #0、1、2等文件夹的路径
    if os.path.isdir(item_path):
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem) #subitem_path是图片的路径
            tp = Image.open(subitem_path)
            tp.rotate(20, expand=True).save(subitem_path)  #旋转20度