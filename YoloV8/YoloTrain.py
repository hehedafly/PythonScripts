from ultralytics import YOLO
import torch
import os
from torch.utils.data import DataLoader
 
# 训练我自己的数据集合
# model = YOLO("yoloTrain.yaml")  # 从头开始构建新模型
model = YOLO("models/yolo11n.pt")  # 加载预训练模型（建议用于训练）
_epochs = 20

def checkWeight(path:str) -> bool:
    return any(file.endswith(".pt") for _, _, files in os.walk(path) for file in files)

# 使用模型
if __name__ == "__main__":
    model.train(data="yoloTrain.yaml", epochs= _epochs, multi_scale=True, profile=True)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    train_dirs = [d for d in os.listdir("./runs/detect") if os.path.isdir(os.path.join("./runs/detect", d)) and d.startswith("train") and checkWeight(os.path.join("./runs/detect", d))]
    train_dirs.sort(key=lambda x: int(x[len("train"):]))

    model = YOLO(f"./runs/detect/{train_dirs[-1]}/weights/best.pt")
    model.export(format="onnx", imgsz = [480, 640],  nms = True, device = "cpu") 
    model.export(format="openvino", imgsz = [480, 640], device = "cpu", batch = 32)
 
# 测试集所在文件夹路径，即pre文件夹
# results = model.predict(source="D:\yolov8教学数据集\pre",save=True,save_conf=True,save_txt=True,name='output')