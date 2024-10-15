# 加载预训练模型
from ultralytics import YOLO
from collections import defaultdict
from pypylon import pylon
import numpy as np
import keyboard
import datetime
import time
import cv2

from IPC import IPCTest

VideoTest = False
resolution = [1440,1080]
# 连接Basler相机列表的第一个相机
if not VideoTest:
    cap = None

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    
    # 开始读取图像
    camera.Open()
    camera.Width.Value = resolution[0]
    camera.Height.Value = resolution[1]
    # camera.PixelFormat = "BGR8"
    camera.Gain.Value = 10
    camera.ExposureTime.Value = 15000
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
else:
    camera = None
    video_path = "Basler_acA1440-220um__40306945__20241005_210252740.mp4"
    cap = cv2.VideoCapture(video_path)

timestr = datetime.datetime.now().strftime("%m_%d_%H%M")

model = YOLO("11sbest.pt")
# video_path = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(timestr+'output.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
outRaw = cv2.VideoWriter(timestr+'outputraw.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
UnityShm = IPCTest.SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject", 32+5*16*1024)#~80KB
UnityShm.InitBuffer()

frame_rate_divider = 1  # 设置帧率除数
frame_count = 0  # 初始化帧计数器
hide = False
selectPlace: list[int] = [-1, -1, -1, -1]
 
counts = defaultdict(int)
object_str = ""
index = 0
startTime = -1
while VideoTest or camera.IsGrabbing():
    if not VideoTest:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
        rectedFrame = frame.copy()
        ret =  grabResult.GrabSucceeded()
    else:
        ret, frame = cap.read()

    outRaw.write(frame)

    if startTime < 0:
        startTime = time.time()
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
                rectedFrame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255,255,0), 2)
                if UnityShm.CheckOnlineClientsCount() > 0:
                    # UnityShm.WriteContent("pos" + ";".join([str(i) for i in xyxy]), True)
                    UnityShm.WriteClear()
                    UnityShm.WriteContent("pos:" + ";".join([str(i) for i in xyxy]))
                    UnityShm.WriteContent("select:" + ";".join([str(i) for i in selectPlace]))
 
        # object_str = object_str +". " + key
        # for class_id, count in counts.items():
        #     object_str = object_str +f"{count} {class_id},"  
        #     counts = defaultdict(int)  
        if selectPlace[0] != -1:
            rectedFrame = cv2.rectangle(rectedFrame, (selectPlace[0], selectPlace[1]), (selectPlace[2], selectPlace[3]), (0,0,255), 2)
        out.write(rectedFrame)
        if not hide:
            cv2.imshow("frame", rectedFrame)
        cv2.waitKey(1)
        # print(key)
        if keyboard.is_pressed("h"):
            hide = not hide
        elif keyboard.is_pressed("v"):
            UnityShm.ShowAllData()
        elif keyboard.is_pressed("s"):
            gROI = cv2.selectROI("ROI frame", frame, False)
            selectPlace = [gROI[0], gROI[1], gROI[0] + gROI[2], gROI[1] + gROI[3]]
            cv2.destroyWindow("ROI frame")
        elif keyboard.is_pressed("esc"):
            break
        elif keyboard.is_pressed("space") and not hide:
            cv2.waitKey()
    
 
    if frame_count % 60 == 0:
        costTime = time.time() - startTime
        print(str(60/costTime)+"fps")
        startTime = time.time()
    frame_count += 1  # 更新帧计数器
 
# object_str= object_str.strip(',').strip('.')
# print("reuslt:", object_str)
 
if camera != None:
    camera.Close()
out.release()
outRaw.release()
cv2.destroyAllWindows()
del UnityShm