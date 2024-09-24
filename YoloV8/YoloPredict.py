# 加载预训练模型
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
import cv2
 
model = YOLO("best.pt")
video_path = "WIN_20240822_15_44_20_Pro.mp4"
# video_path = 0
 
# 打开视频文件
cap = cv2.VideoCapture(video_path)
 
frame_rate_divider = 1  # 设置帧率除数
frame_count = 0  # 初始化帧计数器
 
counts = defaultdict(int)
object_str = ""
index = 0
startTime = -1
while cap.isOpened(): # 检查视频文件是否成功打开
    if startTime < 0:
        startTime = time.time()
    ret, frame = cap.read() # 读取视频文件中的下一帧,ret 是一个布尔值，如果读取帧成功
    if not ret:
        break
 
    # 每隔 frame_rate_divider 帧进行一次预测
    if frame_count % frame_rate_divider == 0:
        results = model(frame, verbose=False)
        
        key = f"({index}): "
        index = index +1
        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                counts[class_id] += 1
                xyxy = np.array(box.xyxy[0].tolist(), int)
                frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255,255,0), 2)
        
 
        object_str = object_str +". " + key
        for class_id, count in counts.items():
            object_str = object_str +f"{count} {class_id},"  
            counts = defaultdict(int)  
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    
 
    if frame_count % 60 == 0:
        costTime = time.time() - startTime
        print(str(60/costTime)+"fps")
        startTime = time.time()
    frame_count += 1  # 更新帧计数器
 
object_str= object_str.strip(',').strip('.')
print("reuslt:", object_str)
 
cap.release()
cv2.destroyAllWindows()