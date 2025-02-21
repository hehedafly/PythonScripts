import datetime
import math
import os
import threading
import time
import copy

# Third-party imports
import torch
import cv2
import keyboard
import numpy as np
from typing import Literal
from collections import defaultdict, deque
from ultralytics import YOLO
from pypylon import pylon
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from IPC import IPCTest
from CircleSelect import CircleSelect
from MessageBox import PyWinMessageBox

# from pyinstrument import Profiler
# region ------------------------------------------------meta Info-------------------------------------------
CameraTypes = ["basler", "common", "video"]
CameraType = "video"
videoPath = "01_17_1842outputraw.mp4"
modelNmae = "models/TopViewbestNew.pt"
confidenceCoefficient = 0.6
UnityshmCare = True
resolution = [1440,1080]
recordPredictResult = False
videoSaveFolder = "outputVideo/"
missedFrameSaveFolder = "missedFrames/"

multiThread = False 
Task: Literal['detect', 'track'] = 'detect'
frame_rate_divider = 1  # 设置帧率除数
frame_count = 0  # 初始化帧计数器
hide = False
simulate = False
selectAreas:list[list[int]] = []
simulateMousePos = [-1, -1, -1, -1, -1]
selectChanged = False
FontSize = 0.8
FontThick = 2
costTime = 0
useCuda = torch.cuda.is_available()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# endregion ------------------------------------------------meta Info end-------------------------------------------

# region ------------------------------------------------camera and log Info-------------------------------------------
# 连接Basler相机列表的第一个相机
if multiThread:
    camera = None
else:
    if CameraType == "basler":
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # 开始读取图像
        camera.Open()
        camera.Width.Value = resolution[0]
        camera.Height.Value = resolution[1]
        # camera.PixelFormat = "BGR8"
        camera.Gain.Value = 7.5
        camera.ExposureTime.Value = 10000
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # mediaNamePure = "Basler" + datetime.datetime.now().strftime("%m_%d_%H%M")
    elif CameraType == "video":
        video_path = videoPath
        camera = cv2.VideoCapture(video_path)

        # mediaNamePure = ".".join(video_path.split(".")[0:-1])
    elif CameraType == "common":
        camera = cv2.VideoCapture(0)
        # mediaNamePure = "camera" + datetime.datetime.now().strftime("%m_%d_%H%M")
    else:
        print("wrong camera type")
        exit()

timestr = datetime.datetime.now().strftime("%m_%d_%H%M")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if recordPredictResult:
    out = cv2.VideoWriter(videoSaveFolder + timestr+'output.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
    if not multiThread:
        outRaw = cv2.VideoWriter(videoSaveFolder + timestr+'outputraw.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
else:
    out = None
    outRaw = None
defineCircle = CircleSelect.DefineCircle()


class FrameGrabber(threading.Thread):
    def __init__(self, _cameraType, fps=50):
        super().__init__()

        if CameraType == "basler":
            self.tempCamera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            # 开始读取图像
            self.tempCamera.Open()
            self.tempCamera.Width.Value = resolution[0]
            self.tempCamera.Height.Value = resolution[1]
  
            self.tempCamera.Gain.Value = 7.5
            self.tempCamera.ExposureTime.Value = 10000
            self.tempCamera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        elif CameraType == "video":
            video_path = videoPath
            camera = cv2.VideoCapture(video_path)

        elif CameraType == "common":
            camera = cv2.VideoCapture(0)

        else:
            print("wrong camera type")
            exit()

        self.cameraType = _cameraType
        self.baslerCamera = self.tempCamera if _cameraType == "basler" else None
        self.camera = camera if _cameraType == "common" or _cameraType == "video" else None

        self.timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = None
        
        self.record = False
        self.fps = fps
        self.recorded_frames = 0
        self.frame_buffer = deque(maxlen=int(fps * 0.2))  # 帧缓存队列
        self.lock = threading.Lock()
        self.running = False
        
        # 动态调整参数
        self.interval = 0.95 * (1.0 / fps)  # 基础间隔缩短5%
        self.adjustment_factor = 0.2         # 延迟补偿系数
        self.last_delay = 0.0

    def cameraRelease(self):
        if self.baslerCamera is not None:
            self.baslerCamera.StopGrabbing()
            self.baslerCamera.Close()
        elif self.camera is not None:
            self.camera.release()

    def getFrame(self) -> tuple[bool, np.ndarray]:
        if self.cameraType == "basler":
            grabResult = self.baslerCamera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
            ret =  grabResult.GrabSucceeded()
            if not ret:
                print("lost connection to basler")
        else:
            ret, frame = self.camera.read()
            if not ret:
                print("no camera connected")
        return ret, frame

    def run(self):
        self.running = True
        next_time = time.time()
        
        while self.running:
            # 动态调整间隔
            adjusted_interval = max(0, self.interval - self.last_delay * self.adjustment_factor)
            
            start_time = time.time()
            ret, frame = self.getFrame()
            if not ret:
                self.running = False
                print("failed to get frame")
                break
            
            # 写入视频
            if self.writer != None and self.record:
                self.writer.write(frame)
            self.recorded_frames += 1 
            
            # 更新缓存
            with self.lock:
                self.frame_buffer.append(frame)
            
            # 计算延迟补偿
            process_time = time.time() - start_time
            actual_delay = process_time - adjusted_interval
            self.last_delay = max(0, actual_delay)
            
            # 控制帧率
            next_time += adjusted_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cameraRelease()
        if self.writer != None:
            self.writer.release()

    # def getFrameCount(self):
    #     return len(self.frame_buffer)

    def get_last_frame(self) -> tuple[bool, int, np.ndarray]:
        with self.lock:
            if len(self.frame_buffer):
                # 返回最新帧并保留缓存
                return True,self.recorded_frames, self.frame_buffer[-1]
            return False, -1, None
        
    def returnCameraStatus(self):
        if self.baslerCamera is not None:
            return self.baslerCamera.IsGrabbing()
        elif self.camera is not None:
            return self.camera.isOpened()

    # def clear_buffer(self):
    #     with self.lock:
    #         self.frame_buffer.clear()

    def startRecord(self):
        self.writer = cv2.VideoWriter(timestr+'outputraw.mp4', fourcc, 60.0, (resolution[0], resolution[1]))
        self.record = True

    def stop(self):
        self.running = False


def WriteFrame(frame, isRaw:bool):
    if isRaw:
        outRaw.write(frame)
    else:
        out.write(frame)

#endregion------------------------------------------------camera and log Info end-------------------------------------------

# region ----------------------------------------model load and predict function-----------------------------------
# model = YOLO("TopViewbest.pt")
class Model():
    def __init__(self, modelName:str):
        self.modelName = modelName
        self.modelType = ""
        self.model = None
        # self.openVINOModel:CompiledModel = None

        if modelName.endswith(".pt"):
            self.model = YOLO(self.modelName)
            self.modelType = "yolo"
        elif modelName.endswith(".onnx"):
            self.model = YOLO(modelName)
            self.modelType = "onnx"
        else:
            print("目前仅支持yolo(.pt)模型，onnx模型和openVINO模型")
            print("目前仅支持yolo(.pt)模型，onnx模型")
            # print("目前仅支持yolo(.pt)模型，openVINO模型")
            exit()

    def Predict(self, img:np.ndarray, task: Literal['detect', 'track'] = 'detect'):
        if self.modelType == "yolo":
            tempmodel:YOLO = self.model
            if task == 'detect':
                results = tempmodel(img, verbose=False, conf = confidenceCoefficient, device = "cuda:0" if useCuda else "cpu")
            elif task == 'track':
                results = tempmodel.track(img, verbose=False, conf = confidenceCoefficient, device = "cuda:0" if useCuda else "cpu")
            for result in results:
                if len(result.boxes):
                    box = result.boxes[0]
                    xyxy = np.array(box.xyxy[0].tolist(), int)
                    return [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
            else:
                return None
        else:
            resizedimg = cv2.resize(img, (640, 480))
            if self.modelType == "onnx":
                if task == 'detect':
                    results = self.model(resizedimg, imgsz=(480,640), verbose=False, conf = confidenceCoefficient, device = "cuda:0" if useCuda else "cpu")
                elif task == 'track':
                    results = self.model.track(resizedimg, imgsz=(480,640), verbose=False, conf = confidenceCoefficient, device = "cuda:0" if useCuda else "cpu")
                for result in results:
                    if len(result.boxes):
                        box = result.boxes[0]
                        xyxy = np.array(box.xyxy[0].tolist(), int)
                        return [int((xyxy[0]+xyxy[2])*0.5 * (1440/640)), int((xyxy[1]+xyxy[3])*0.5 * (1080/480))]
                else:
                    return None 
            return None

model = Model(modelNmae)
    
UnityShm = IPCTest.SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject" if UnityshmCare else "", 32+5*16*1024)#~80KB
UnityShm.InitBuffer()

# endregion----------------------------------------model load and predict function end-----------------------------------

#selectPlace: list[int] = [-1, -1, -1, -1, -1, -1]#type: mark; type(check pos region), 0-rectange, 1-circle ; x/centerx ; y/centery ; w/rad ; h/inner

# region ----------------------------------------scene and selectAreas info-----------------------------------

startTime = -1
unityFixedUscaledTimeOffset:float = 0
createdTime = time.process_time()
sync = False
syncInd:int = -1
markCountPerType = 32
sceneInfo:list[float] = []#[sceneCenter[0], sceneCenter[1], sceneRadius, sceneAngle]
selectionSaveTxtName = "scene and selectAreas.txt"

f_selectionSaveTxt = open(selectionSaveTxtName, 'r+' if os.path.exists(selectionSaveTxtName) else 'w+', encoding='utf-8')
content = f_selectionSaveTxt.readlines()
sceneContent = [s for s in content if s.startswith("scene:")]
areaContent = [s for s in content if s.startswith("selectAreas:")]
if len(sceneContent) and PyWinMessageBox.YesOrNo("load Previous SceneInfo?", "save & load") == 'YES':
    for line in sceneContent:
        line = line.replace("\n", "")
        sceneInfo = [float(i) for i in (line.split(':')[1]).split(';')]
        print("scene info loaded: " + line)
if len(areaContent) and PyWinMessageBox.YesOrNo("load Previous select areas?", "save & load") == 'YES':
    for line in areaContent:
        line = line.replace("\n", "")
        selectAreas.append([int(i) for i in (line.split(':')[1]).split(';')])
        print("selectAreas info loaded: " + line)


realMouseCenter = [-1, -1]
def ProcessMouseNearRegion(pos, _frame):
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    tempMask = np.zeros_like(_frame)
    tempMask = cv2.circle(tempMask, pos, 80, (255,255,255), 160)
    masked_frame = cv2.bitwise_and(_frame, _frame, mask=tempMask)
    np.tile(masked_frame[:, :, np.newaxis], (1, 1, 3))

    return  np.tile(masked_frame[:, :, np.newaxis], (1, 1, 3))

def PointOffset(point, offset:int):
    return (point[0] + offset, point[1] - offset)

def drawSelectArea(frame, selectAreas:list[list[int]], color = None):
    global markCountPerType
    fontSize = 1.5
    fontThick = 2
    
    for selectPlace in selectAreas:
        mark = selectPlace[0] // markCountPerType
        if color == None:
            drawcolor = (255, 0, 0) if mark == 0 else (0, 0, 255)
        else:
            drawcolor = color

        if selectPlace[1] == 0:
            frame = cv2.circle(frame, (selectPlace[2], selectPlace[3]), selectPlace[4], drawcolor, 2)
            frame = cv2.putText(frame, str(selectPlace[0] % markCountPerType), PointOffset((selectPlace[2], selectPlace[3]), -10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, drawcolor, fontThick)

        elif selectPlace[1] == 1:
            frame = cv2.rectangle(frame, (selectPlace[2], selectPlace[3]), (selectPlace[4], selectPlace[5]), drawcolor, 2)
            frame = cv2.putText(frame, str(selectPlace[0] % markCountPerType), PointOffset(((selectPlace[2] + selectPlace[4]) // 2, (selectPlace[3] + selectPlace[5]) // 2), -10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, drawcolor, fontThick)

# endregion ----------------------------------------scene and selectAreas info-----------------------------------

#region Matplotlib

class GUI:
    def __init__(self, frame, selectList:list[list[int]], sceneInfo:list[float]):
        # cv2.imshow("__init__ origin frame: ", frame)
        self.oframe = copy.deepcopy(frame)
        self.frame = copy.deepcopy(frame)
        self.sceneInfo = sceneInfo
        (h ,w, _) = frame.shape
        self.selectListRef = selectList
        self.oselectList = copy.deepcopy(selectList)
        self.selectList = copy.deepcopy(selectList)
        self.type0List = [item for item in selectList if len(item) >= 0 and 0 <= item[0] < markCountPerType]
        self.type1List = [item for item in selectList if len(item) >= 0 and markCountPerType <= item[0] < markCountPerType*2]
        # 创建GUI
        self.fig, self.ax = plt.subplots(figsize=(w * 0.008, h * 0.008 + 1))
        self.buttonWidth = 0.1
        self.buttonHeight = 0.04
        # tempAxPos = self.ax.get_position().get_points()
        # tempAxPos = np.append(tempAxPos[0], tempAxPos[1])
        self.ax.set_position([0.04, self.buttonWidth, 0.94, 0.98])

        self.DrawAllContent()

        self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        self.createButton()

        plt.show()

    def DrawAllContent(self):
        sceneCenter = (int(sceneInfo[0]), int(sceneInfo[1]))
        sceneRadius = int(sceneInfo[2])
        sceneAngle = sceneInfo[3]
        self.frame = cv2.circle(self.frame, sceneCenter, sceneRadius, (0, 255, 0), 2)
        self.frame = CircleSelect.draw_arrow(self.frame, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
        
        drawSelectArea(self.frame, self.selectList)

    def createButton(self):
        
        self.button0circle_ax = plt.axes([0.1, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.button0circle = widgets.Button(self.button0circle_ax, 'Trigger region\n circle ')
        self.button0circle.on_clicked(self.add_to_list)

        self.button0rect_ax = plt.axes([0.1, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.button0rect = widgets.Button(self.button0rect_ax, 'Trigger region\n rect ')
        self.button0rect.on_clicked(self.add_to_list)

        self.button1circle_ax = plt.axes([0.25, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.button1circle = widgets.Button(self.button1circle_ax, 'Destination\n circle ')
        self.button1circle.on_clicked(self.add_to_list)

        self.button1rect_ax = plt.axes([0.25, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.button1rect = widgets.Button(self.button1rect_ax, 'Destination\n rect ')
        self.button1rect.on_clicked(self.add_to_list)

        self.circleArray_ax = plt.axes([0.4, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.circleArray_button = widgets.Button(self.circleArray_ax, 'Circle Array\n Last Selection')
        self.circleArray_button.on_clicked(self.CircleArray)

        self.pop_ax = plt.axes([0.55, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.pop_button = widgets.Button(self.pop_ax, 'Pop')
        self.pop_button.on_clicked(self.ClearList)

        self.clear_ax = plt.axes([0.55, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.clear_button = widgets.Button(self.clear_ax, 'Clear')
        self.clear_button.on_clicked(self.ClearList)

        self.close_ax = plt.axes([0.7, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.close_button = widgets.Button(self.close_ax, 'Finish')
        self.close_button.on_clicked(self.Close_gui)

    def Init(self, _oframe):
        # cv2.imshow("__init__ origin frame: ", _oframe)

        self.frame = copy.deepcopy(_oframe)
        (h ,w, _) = self.frame.shape
        self.type0List = [item for item in self.selectList if len(item) >= 0 and 0 <= item[0] < markCountPerType]
        self.type1List = [item for item in self.selectList if len(item) >= 0 and markCountPerType <= item[0] < markCountPerType*2]
        # 创建GUI
        self.fig, self.ax = plt.subplots(figsize=(w * 0.008, h * 0.008 + 1))

        self.ax.set_position([0.04, self.buttonWidth, 0.94, 0.98])

        self.DrawAllContent()
        
        self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        self.createButton()

        plt.show()

    def ClearList(self, event):
        clearAll = True if event.inaxes == self.clear_ax else False
        if len(self.selectList):
            if clearAll:
                self.selectList.clear()
            else:
                self.selectList = self.selectList[:-1]
            plt.close(self.fig)
            self.Init(self.oframe)
    
    def CircleArray(self, event):#仅支持圆形环形阵列
        if len(self.selectList):
            last:list[int] = self.selectList[-1]
            if last[1] != 0:

                return

            mark = last[0]
            markTypeMax = (mark // markCountPerType + 1) * markCountPerType
            center_x = self.sceneInfo[0]
            center_y = self.sceneInfo[1]
            
            original_center_x = last[2]
            original_center_y = last[3]
            oroginal_angle = math.atan2(original_center_y - center_y, original_center_x - center_x)
            radius = last[4]
            inner = last[5]
            
            # 计算环的半径（形状中心到旋转中心的距离）
            ring_radius = math.hypot(original_center_x - center_x, original_center_y - center_y)
            
            for i in range(7):
                if mark + i < markTypeMax:
                    angle_deg = (i + 1) * 360 / 8  # degrees
                    angle_rad = math.radians(angle_deg) + oroginal_angle
                    # 计算新的圆心坐标
                    new_center_x = center_x + ring_radius * math.cos(angle_rad)
                    new_center_y = center_y + ring_radius * math.sin(angle_rad)
                    # 保持半径不变
                    new_shape = [mark + i + 1, 0, int(new_center_x + 0.5), int(new_center_y + 0.5), radius, inner]
                    self.selectList.append(new_shape)
            plt.close(self.fig)
            self.Init(self.oframe)

    def add_to_list(self, event):
        selectPlace: list[int] = [-1, -1, -1, -1, -1, -1]
        # self = param
        fig = self.fig
        frame = self.frame
        selectList = self.selectList
        plt.close(fig)
        ButtonLabel = 0 if event.inaxes in [self.button0circle_ax, self.button0rect_ax] else 1
        selectType = 0 if event.inaxes in [self.button0circle_ax, self.button1circle_ax] else 1
        if (ButtonLabel == 0 and len(self.type0List) < markCountPerType) or (ButtonLabel == 1 and len(self.type1List) < markCountPerType):
            if selectType == 0:#circle
                selectPlace[0] = len(self.type0List) if ButtonLabel == 0 else markCountPerType + len(self.type1List)
                selectPlace[1] = 0
                _center, radius, inner = defineCircle.define_circle_by_center_and_point(frame)
                if(_center != None):
                    selectPlace[2:6] = [_center[0], _center[1], radius, 1 if inner else 0]
                
            elif selectType == 1:#rect
                selectPlace[0] = len(self.type0List) if ButtonLabel == 0 else markCountPerType + len(self.type1List)
                selectPlace[1] = 1
                gROI = cv2.selectROI("ROI frame", frame, False)
                selectPlace[2:6] = [gROI[0], gROI[1], gROI[0] + gROI[2], gROI[1] + gROI[3]]
                cv2.destroyWindow("ROI frame")

            selectList.append(selectPlace)
            print("added: [" + ",".join([str(s) for s in selectPlace]) + "]")
        else:{
            print("列表已满")
        }

        self.Init(self.oframe)

    def Close_gui(self, event):
        self.selectListRef.clear()
        for select in self.selectList:
            self.selectListRef.append(select)
        plt.close(self.fig)

#endregion

def getFrame() -> tuple[bool, np.ndarray, int]:
    if multiThread:
        ret, frameInd, frame = grabber.get_last_frame()
        # grabber.clear_buffer()
        return ret, frame, frameInd
    else:
        if CameraType == "basler":
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
            ret =  grabResult.GrabSucceeded()
            if not ret:
                print("lost connection to basler")
        else:
            ret, frame = camera.read()
            if not ret:
                print("no camera connected")
        return ret, frame, -1
    
def TrygetFrame(waitTime:float = 0.01) -> tuple[bool, np.ndarray, int]:
    timer:list[float] = [time.time(), -1]
    while True:
        ret, frame, frameInd = getFrame()
        if ret:
            return ret, frame, frameInd
        else:
            if timer[1] == -1:
                timer[1] = time.time()
                time.sleep(waitTime * 0.1)
            if time.time() - timer[1] > waitTime:
                return ret, frame, frameInd


if multiThread:
    grabber = FrameGrabber(_cameraType= CameraType)
    grabber.start()

def Quit():
    if multiThread:
        grabber.stop()
    quit()

fristFrame = None
selectSceneMask = None
selectMask = None
PreBoolMask = None
if len(sceneInfo) == 0:
    ret, fristFrame, frameInd = TrygetFrame(0.1)
    if not ret:
        print("no camera connected")
        Quit()
    selectSceneMask = np.zeros_like(fristFrame)
    selectMask = np.zeros_like(fristFrame)

    sceneCenter, sceneRadius, sceneAngle = defineCircle.define_circle_by_three_points(fristFrame)
    # sceneAngle -= 90 #defineCircle中以正上方为0
    sceneInfo = [sceneCenter[0], sceneCenter[1], sceneRadius, sceneAngle]
    selectChanged = True
    selectSceneMask = cv2.circle(selectSceneMask, sceneCenter, sceneRadius, (0, 255, 0), 2)
    selectSceneMask = CircleSelect.draw_arrow(selectSceneMask, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)

else:
    ret, fristFrame, _ = TrygetFrame(0.1)
    if not ret:
        print("no camera connected")
        Quit()
    selectSceneMask = np.zeros_like(fristFrame)
    selectMask = np.zeros_like(fristFrame)

    sceneCenter = (int(sceneInfo[0]), int(sceneInfo[1]))
    sceneRadius = int(sceneInfo[2])
    sceneAngle = sceneInfo[3]
    selectSceneMask = cv2.circle(selectSceneMask, sceneCenter, sceneRadius, (0, 255, 0), 2)
    selectSceneMask = CircleSelect.draw_arrow(selectSceneMask, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
drawSelectArea(selectMask, selectAreas)
PreBoolMask = ~(selectMask.any(axis=-1))
selectMask[PreBoolMask] = selectSceneMask[PreBoolMask]
PreBoolMask = selectMask.any(axis=-1)

availableMask = np.ones(fristFrame.shape, dtype= np.uint8)
tempMask = cv2.imread("tempMask.jpg")
if(type(tempMask) != type(None) and tempMask.shape == fristFrame.shape):
    availableMask = tempMask
else:
    while True:         
        gROI = cv2.selectROI("ROI frame", fristFrame * availableMask, False)

        availableMask[gROI[1]:(gROI[1] + gROI[3]), gROI[0]:(gROI[0] + gROI[2])] = 0

        if keyboard.is_pressed('s'):
            cv2.imwrite("tempMask.jpg", availableMask)
        elif keyboard.is_pressed('esc'):
            cv2.destroyWindow("ROI frame")
            break
            
        if gROI == (0,0,0,0):
            cv2.destroyWindow("ROI frame")
            break

def simulateMousePosUpdate(event, x, y, flags, param):
    simulateMousePos[0:2] = [x, y]

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", simulateMousePosUpdate)

if multiThread and recordPredictResult:
    grabber.startRecord()

ProcessStartTime = time.time()

# profiler = Profiler()
# profiler.start()

while CameraType != "basler" or (multiThread or camera.IsGrabbing()):
    ret, frame, frameInd = getFrame()
    if not ret:
        break
    rectedFrame = copy.deepcopy(frame)

    if recordPredictResult:
        if not multiThread:
            WriteFrame(frame, True)
        outWritten:bool = False

    # if realMouseCenter[0] > 0:
    #     Predictframe = ProcessMouseNearRegion(realMouseCenter, frame)
    # else:
    #     # Predictframe = copy.deepcopy(frame) * availableMask
    #     Predictframe = frame * availableMask
    Predictframe = frame * availableMask

    if startTime < 0:
        startTime = time.time()
    if not ret:
        break

    receiveUnityTimeSuccess = -1
    lastReceiveUnityTime = -1
    onlineNumber = UnityShm.CheckOnlineClientsCount()
    if UnityShm.care != "" :
        if onlineNumber > 0:
            if UnityShm.careindex == -1:
                UnityShm.CheckApplies()
            else:
                if onlineNumber > 0 and not sync:
                    syncTryTimes = 0
                    syncTryTimesMax = 100
                    while(syncTryTimes < syncTryTimesMax):
                        timeMsg = UnityShm.ReadToStr(UnityShm.careindex)
                        timeMsg.reverse()
                        for msg in timeMsg:
                            if msg.startswith("time:"):
                                print("from unity: "+msg)
                                temptime = float(msg[5:])
                                if receiveUnityTimeSuccess < 1:#至少连续接收两次
                                    print("success: "+ str(temptime))

                                    if lastReceiveUnityTime == -1:
                                        lastReceiveUnityTime = temptime
                                        print("lastReceiveUnityTime init: "+ str(lastReceiveUnityTime))

                                    else:
                                        print("lastReceiveUnityTime update: "+ str(lastReceiveUnityTime))
                                        if temptime - lastReceiveUnityTime < 0.05:#顺利接收
                                            receiveUnityTimeSuccess += 1
                                            unityFixedUscaledTimeOffset = time.process_time() - createdTime - temptime
                                            lastReceiveUnityTime = temptime
                                            for i in range(10):
                                                UnityShm.WriteContent("scene:" + f"{sceneCenter[0]};{sceneCenter[1]};{sceneRadius:.2f};{sceneAngle:.2f}")
                                                time.sleep(0.01)
                                            sync = True
                                            syncInd = 0
                                            UnityShm.WriteClear()
                                            for i in range(2):
                                                for selectedAreaSync in selectAreas:
                                                    UnityShm.WriteContent("select:" + ";".join(map(str, selectedAreaSync)))
                                            print("sync succeed")
                                            syncTryTimes = syncTryTimesMax
                                            break

                                        else:
                                            print("interval:" + str(temptime - lastReceiveUnityTime))
                                            receiveUnityTimeSuccess = -1
                                            lastReceiveUnityTime = -1
                                            syncInd = -1
                                    break

                                else :
                                    if abs(unityFixedUscaledTimeOffset - (time.process_time() - createdTime - temptime)) > 0.5:
                                        print("still too lag")
                                        receiveUnityTimeSuccess = -1
                                        lastReceiveUnityTime = -1
                                        syncInd = -1
                        UnityShm.WriteContent(f"time:{time.process_time() - createdTime}")
                        syncTryTimes += 1
                        time.sleep(0.01)
                    # UnityShm.ReadToStr(1)
                # UnityShm.WriteContent("scene:" + f"{sceneCenter[0]};{sceneCenter[1]};{sceneRadius:.2f};{sceneAngle:.2f}")
            # else:
            #     UnityShm.WriteContent(f"time:{time.process_time() - createdTime}")
            #     UnityShm.ReadToStr(1)
        elif onlineNumber == 0:
            if sync:
                print("0 online member")
            receiveUnityTimeSuccess = -1
            unityFixedUscaledTimeOffset = 0
            sync = False
        else:#sync = true
            readMsg = UnityShm.ReadToStr(1)

    UnityShmPrepared:bool = (UnityShm.care != "" and sync) or UnityShm.care == ""

    if frame_count % 30 == 0:
        costTime = time.time() - startTime
        # print(str(60/costTime)+"fps")
        startTime = time.time()
    if costTime > 0:
        cv2.putText(rectedFrame, f"fps: {(30/costTime):.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, FontSize, (0, 0, 255), FontThick)

    if frame_count % frame_rate_divider == 0:
        # if UnityShmPrepared:
        if UnityShmPrepared:
            results = model.Predict(Predictframe, task= Task)
        else:
            results = []
        syncInd += 1 if syncInd >= 0 else 0
        if simulate:
            simulateMousePos[2] = syncInd if syncInd >= 0 else -1
            simulateMousePos[3] = int((time.time() - ProcessStartTime) * 100)
            simulateMousePos[4] = frameInd
            # print(simulateMousePos)
            UnityShm.WriteContent("pos:" + ";".join([str(i) for i in simulateMousePos]))
        # index = index +1
        rectedFrame[PreBoolMask] = selectMask[PreBoolMask]

        # for result in results:
        #     for box in result.boxes:
        #         class_id = result.names[box.cls[0].item()]
        #         # if class_id == 0:
        #         xyxy = np.array(box.xyxy[0].tolist(), int)
        #         realMouseCenter = [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
        if type(results) != type(None) and len(results) == 2:
            realMouseCenter = results
            rectedFrame = cv2.circle(rectedFrame, realMouseCenter, 5, (255,255,0), 10)
            if not simulate:
                # UnityShm.WriteContent("pos" + ";".join([str(i) for i in xyxy]), True)
                # UnityShm.WriteClear()
                temp = realMouseCenter.copy()
                temp.append(syncInd if syncInd >= 0 else -1)
                temp.append(int((time.time() - ProcessStartTime) * 100))
                temp.append(frameInd)
                simulateMousePos[2] = syncInd if syncInd >= 0 else -1
                UnityShm.WriteContent("pos:" + ";".join([str(i) for i in temp]))

            # break
        else:
            realMouseCenter = [-1, -1]
            cv2.imwrite(missedFrameSaveFolder + timestr + f"frame{frame_count}.jpg")

        if recordPredictResult:
            out.write(rectedFrame)
            outWritten = True

        if not hide:
            cv2.imshow("frame", rectedFrame)
        cv2.waitKey(1)

        if keyboard.is_pressed("shift+h"):
            hide = not hide
            while keyboard.is_pressed("shift+h"):
                continue
        elif keyboard.is_pressed("shift+v"):
            if UnityShmPrepared:
                UnityShm.ShowAllData()
                while keyboard.is_pressed("shift+v"):
                    continue
        elif keyboard.is_pressed("shift+s"):
            gui = GUI(frame, selectAreas, sceneInfo)

            selectMask.fill(0)
            drawSelectArea(selectMask, selectAreas)
            PreBoolMask = ~(selectMask.any(axis=-1))
            selectMask[PreBoolMask] = selectSceneMask[PreBoolMask]
            PreBoolMask = selectMask.any(axis=-1)

            oselect_set = set(map(tuple, gui.oselectList))
            select_set = set(map(tuple, selectAreas))

            added = select_set - oselect_set
            deleted = oselect_set - select_set
            if len(added) != 0 or len(deleted) != 0:
                selectChanged = True
            if UnityShmPrepared:
                for item in added:
                    UnityShm.WriteContent("select:" + ";".join(map(str, item)))

                # 处理被删除的内容
                for item in deleted:
                    item_list = list(item)
                    item_list[0] = (item_list[0] + 1) * -1
                    UnityShm.WriteContent("select:" + ";".join(map(str, item_list)))

        elif keyboard.is_pressed("shift+esc"):
            if not sync or PyWinMessageBox.YesOrNo("Unity Project still online, force exit?", "Warning") == "YES":
                break
        elif keyboard.is_pressed("shift+space") and not hide:
            cv2.waitKey()
        elif keyboard.is_pressed("shift+m"):
            simulate = not simulate
            print("simulate " + ("on" if simulate else "off"))
            while keyboard.is_pressed("shift+m"):
                continue

    if recordPredictResult and not outWritten:
        out.write(frame)

    frame_count += 1

# profiler.stop()
# profiler.print()

if selectChanged and PyWinMessageBox.YesOrNo("save current Info?", "save & load") == 'YES':
    f_selectionSaveTxt.seek(0, 0)
    f_selectionSaveTxt.truncate(0)
    f_selectionSaveTxt.write("scene:"+";".join([str(i) for i in sceneInfo]) + "\n")
    for selectArea in selectAreas:
        f_selectionSaveTxt.write("selectAreas:"+";".join([str(i) for i in selectArea]) + "\n")

f_selectionSaveTxt.close()

if multiThread:
    grabber.stop()
elif camera != None:
    if CameraType == "basler":
        camera.StopGrabbing()
        camera.Close()
    else:
        camera.release()

if recordPredictResult and out != None:
    out.release()
    if not multiThread and outRaw != None:
        outRaw.release()
    print("video stream released")
cv2.destroyAllWindows()
del UnityShm