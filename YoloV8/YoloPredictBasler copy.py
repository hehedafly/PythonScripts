# 加载预训练模型
from ultralytics import YOLO
from collections import defaultdict
from pypylon import pylon
import numpy as np
import keyboard
import datetime
import ctypes
import time
import cv2

# from IPC import IPCTest

# user32 = ctypes.windll.user32

CameraTypes = ["basler", "common", "video"]
CameraType = "video"
resolution = [1440,1080]
# 连接Basler相机列表的第一个相机
if CameraType == "basler":
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
elif CameraType == "video":
    camera = None
    video_path = "RunBetween2RandomPos.mp4"
    cap = cv2.VideoCapture(video_path)
elif CameraType == "common":
    camera = None
    cap = cv2.VideoCapture(0)
else:
    print("wrong camera type")
    exit()

# timestr = datetime.datetime.now().strftime("%m_%d_%H%M")

# model = YOLO("11nbest.pt").track
# # video_path = 0
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(timestr+'output.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
# outRaw = cv2.VideoWriter(timestr+'outputraw.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
# UnityShm = IPCTest.SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject", 32+5*16*1024)#~80KB
# UnityShm.InitBuffer()

# frame_rate_divider = 1  # 设置帧率除数
# frame_count = 0  # 初始化帧计数器
# hide = False
# selectPlace: list[int] = [-1, -1, -1, -1, -1]#type: 0-rectange, 1-circle ; lu/centerx ; lb/centery ; ru/rad ; rb/- ; soft(0-255)
 
# counts = defaultdict(int)
# object_str = ""
# index = 0
# startTime = -1
# while CameraType or camera.IsGrabbing():
if CameraType == "basler":
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
    rectedFrame = frame.copy()
    ret =  grabResult.GrabSucceeded()
else:
    ret, frame = cap.read()
    rectedFrame = frame.copy()

bkgInit = cv2.imread("RunBetween2RandomPosBkg.bmp")
frame_gray_init = cv2.cvtColor(bkgInit, cv2.COLOR_BGR2GRAY)

hsv_canvas = np.zeros_like(frame)
# set saturation value (position 2 in HSV space) to 255
hsv_canvas[..., 1] = 255

subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=10, detectShadows=False)
subtractor.apply(bkgInit)

#     outRaw.write(frame)
#     outWritten:bool = False

#     if startTime < 0:
#         startTime = time.time()
#     if not ret:
#         break

#     if frame_count % frame_rate_divider == 0:
#         results = model(frame, verbose=False)
        
#         key = f"({index}): "
#         index = index +1
#         for result in results:
#             for box in result.boxes:
#                 class_id = result.names[box.cls[0].item()]
#                 # if class_id == 0:
#                 xyxy = np.array(box.xyxy[0].tolist(), int)
#                 center = [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
#                 rectedFrame = cv2.circle(frame, center, 8, (255,255,0), 16)
#                 # rectedFrame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255,255,0), 2)
#                 if UnityShm.CheckOnlineClientsCount() > 0:
#                     # UnityShm.WriteContent("pos" + ";".join([str(i) for i in xyxy]), True)
#                     UnityShm.WriteClear()
#                     UnityShm.WriteContent("pos:" + ";".join([str(i) for i in center]))
#                     UnityShm.WriteContent("select:" + ";".join([str(i) for i in selectPlace]))
 
#         # object_str = object_str +". " + key
#         # for class_id, count in counts.items():
#         #     object_str = object_str +f"{count} {class_id},"  
#         #     counts = defaultdict(int)  
#         if selectPlace[0] == 0:
#             rectedFrame = cv2.rectangle(rectedFrame, (selectPlace[0], selectPlace[1]), (selectPlace[2], selectPlace[3]), (0,0,255), 2)
#         elif selectPlace[0] == 1:
#             rectedFrame = cv2.circle(rectedFrame, (selectPlace[0], selectPlace[1]), (selectPlace[2], selectPlace[3]), (0,0,255), 2)
#         out.write(rectedFrame)
#         outWritten = True
#         if not hide:
#             cv2.imshow("frame", rectedFrame)
#         cv2.waitKey(1)
#         # print(key)
#         # print(user32.GetForegroundWindow())
#         # print(cv2.getWindowProperty("frame", cv2.WND_PROP_AUTOSIZE))
#         # if user32.GetForegroundWindow() == cv2.getWindowProperty("frame", cv2.WND_PROP_AUTOSIZE):

#         if keyboard.is_pressed("h"):
#             hide = not hide
#         elif keyboard.is_pressed("v"):
#             UnityShm.ShowAllData()
#         elif keyboard.is_pressed("s"):
#             gROI = cv2.selectROI("ROI frame", frame, False)
#             selectPlace = [gROI[0], gROI[1], gROI[0] + gROI[2], gROI[1] + gROI[3]]
#             cv2.destroyWindow("ROI frame")
#         elif keyboard.is_pressed("esc"):
#             break
#         elif keyboard.is_pressed("space") and not hide:
#             cv2.waitKey()
    
#     if not outWritten:
#         out.write(frame)
        
#     if frame_count % 60 == 0:
#         costTime = time.time() - startTime
#         print(str(60/costTime)+"fps")
#         startTime = time.time()
#     frame_count += 1
 
# # object_str= object_str.strip(',').strip('.')
# # print("reuslt:", object_str)
 
# if camera != None:
#     camera.Close()
# out.release()
# outRaw.release()
# cv2.destroyAllWindows()
# del UnityShm

while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break
    fg_mask = subtractor.apply(frame)

    # 进行形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations = 5)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个副本用于绘制
    contour_image = fg_mask.copy()

    # 绘制并填充轮廓
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
    # fg_mask = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel, iterations = 3)

    _, inv_mask = cv2.threshold(fg_mask, 1, 255, cv2.THRESH_BINARY_INV)

    white_image = 255 * np.ones_like(frame, dtype=np.uint8)

    # 使用掩码的逆将原始帧的未遮罩区域设置为白色
    # 首先，将掩码应用于原始帧，保留遮罩区域的颜色
    masked_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # 将白色图像应用到未遮罩区域
    unmasked_frame = cv2.bitwise_and(white_image, white_image, mask=inv_mask)

    # 将两个结果合并
    result_frame = cv2.bitwise_or(masked_frame, unmasked_frame)


    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # compare initial frame with current frame
    # flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    # # get x and y coordinates
    # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # # set hue of HSV canvas (position 1)
    # hsv_canvas[..., 0] = angle*(180/(np.pi/2))
    # # set pixel intensity value (position 3
    # hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)

    # # optional recording result/mask
    # # video_output.write(frame_rgb)

    # cv2.imshow('Optical Flow (dense)', frame_rgb)
    # frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    cv2.imshow("mask", fg_mask)
    cv2.imshow("frame", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # set initial frame to current frame