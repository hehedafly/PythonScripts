# coding:utf-8
 
import os
import shutil
import random
import argparse
import numpy as np
 
parser = argparse.ArgumentParser()
parser.add_argument('--rawDataPath', default='D:/Unity/PythonFiles/YoloV8/OutputMouseBodyPic0116044231', type=str, help='All File path')
# 数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='D:/Unity/PythonFiles/YoloV8/YoloTrainData/labels', type=str, help='output txt label path')
parser.add_argument('--img_path', default='D:/Unity/PythonFiles/YoloV8/YoloTrainData/images', type=str, help='output img path')
opt = parser.parse_args()
 
trainval_percent = 0.9
train_percent = 0.9
RawDataPath = opt.rawDataPath
txtsavepath = opt.txt_path
imgsavepath = opt.img_path

totalFile = []
with os.scandir(RawDataPath) as entries:
        for entry in entries:
            if entry.name.endswith(".png"):
            # if not entry.name.endswith(".png"):
                totalFile.append(entry.name[:-7])

dirs = [txtsavepath+"/train", txtsavepath+"/val", txtsavepath+"/test", imgsavepath+"/train", imgsavepath+"/val", imgsavepath+"/test"]
for path in dirs:
    if not os.path.exists(path):
        os.makedirs(path)
 
num = len(totalFile)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)
 
# file_trainval = open(txtsavepath + '/trainval.txt', 'w')
# file_test = open(txtsavepath + '/test.txt', 'w')
# file_train = open(txtsavepath + '/train.txt', 'w')
# file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = totalFile[i]
    if i in trainval:
        # file_trainval.write(name)
        if i in train:
            shutil.copy(RawDataPath+"/"+name+".jpg", dirs[3]+"/"+name+".jpg")
            shutil.copy(RawDataPath+"/"+name+".txt", dirs[0]+"/"+name+".txt")
            # file_train.write(name)
        else:
            shutil.copy(RawDataPath+"/"+name+".jpg", dirs[4]+"/"+name+".jpg")
            shutil.copy(RawDataPath+"/"+name+".txt", dirs[1]+"/"+name+".txt")
            # file_val.write(name)
    else:
        shutil.copy(RawDataPath+"/"+name+".jpg", dirs[5]+"/"+name+".jpg")
        shutil.copy(RawDataPath+"/"+name+".txt", dirs[2]+"/"+name+".txt")
        # file_test.write(name)
 
# file_trainval.close()
# file_train.close()
# file_val.close()
# file_test.close()