from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
 
# 训练我自己的数据集合
# model = YOLO("yoloTrain.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）
 
# 使用模型
if __name__ == "__main__":
    model.train(data="yoloTrain.yaml", epochs=10)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
 
# 测试集所在文件夹路径，即pre文件夹
# results = model.predict(source="D:\yolov8教学数据集\pre",save=True,save_conf=True,save_txt=True,name='output')