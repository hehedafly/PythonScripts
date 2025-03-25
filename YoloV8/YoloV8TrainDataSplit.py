# coding:utf-8
 
import os
import shutil
import random
import argparse
import numpy as np

from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
 
parser = argparse.ArgumentParser()
parser.add_argument('--rawDataPath', default='D:/Unity/PythonFiles/YoloV8/OutputMouseBodyPic0322114024', type=str, help='All File path')
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
 
# 训练我自己的数据集合
# model = YOLO("yoloTrain.yaml")  # 从头开始构建新模型
model = YOLO("models/yolo11n.pt")  # 加载预训练模型（建议用于训练）
_epochs = 30

def checkWeight(path:str) -> bool:
    return any(file.endswith(".pt") for _, _, files in os.walk(path) for file in files)

# 使用模型
if __name__ == "__main__":
    model.train(data="yoloTrain.yaml", epochs= _epochs, multi_scale=True, profile=True, workers=0)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    train_dirs = [d for d in os.listdir("./runs/detect") if os.path.isdir(os.path.join("./runs/detect", d)) and d.startswith("train") and checkWeight(os.path.join("./runs/detect", d))]
    train_dirs.sort(key=lambda x: int(x[len("train"):]))

    model = YOLO(f"./runs/detect/{train_dirs[-1]}/weights/best.pt")
    model.export(format="onnx", imgsz = [480, 640],  nms = True, device = "cpu") 
    # model.export(format="openvino", imgsz = [480, 640], device = "cpu", batch = 32)
 