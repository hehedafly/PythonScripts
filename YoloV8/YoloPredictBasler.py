# 加载预训练模型
from ultralytics import YOLO
from collections import defaultdict
from pypylon import pylon
import numpy as np
import matplotlib.widgets as widgets
from matplotlib import pyplot as plt
import keyboard
import datetime
import math
import copy
import time
import cv2
import os

from IPC import IPCTest
from CircleSelect import CircleSelect
from MessageBox import PyWinMessageBox

CameraTypes = ["basler", "common", "video"]
CameraType = "video"
videoPath = "01_15_2012outputraw.mp4"
UnityshmCare = True
resolution = [1440,1080]
recordPredictResult = False

# 连接Basler相机列表的第一个相机
if CameraType == "basler":
    cap = None

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # 开始读取图像
    camera.Open()
    camera.Width.Value = resolution[0]
    camera.Height.Value = resolution[1]
    # camera.PixelFormat = "BGR8"
    camera.Gain.Value = 7.5
    camera.ExposureTime.Value = 10000
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    mediaNamePure = "Basler" + datetime.datetime.now().strftime("%m_%d_%H%M")
elif CameraType == "video":
    camera = None
    video_path = videoPath
    cap = cv2.VideoCapture(video_path)

    mediaNamePure = ".".join(video_path.split(".")[0:-1])
elif CameraType == "common":
    camera = None
    cap = cv2.VideoCapture(0)
    mediaNamePure = "camera" + datetime.datetime.now().strftime("%m_%d_%H%M")
else:
    print("wrong camera type")
    exit()

timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
# tempPicFolderName = "PredicOutputMouseBodyPic" + timestr
# if not os.path.exists(tempPicFolderName):
# 		os.makedirs(tempPicFolderName)

# model = YOLO("11nbest.pt").track
# model = YOLO("11nbottombestSecond.pt")
model = YOLO("TopViewbest.pt")
defineCircle = CircleSelect.DefineCircle()
# video_path = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if recordPredictResult:
    out = cv2.VideoWriter(timestr+'output.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
    outRaw = cv2.VideoWriter(timestr+'outputraw.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
UnityShm = IPCTest.SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject" if UnityshmCare else "", 32+5*16*1024)#~80KB
UnityShm.InitBuffer()

frame_rate_divider = 1  # 设置帧率除数
frame_count = 0  # 初始化帧计数器
hide = False
simulate = False
selectAreas:list[list[int]] = []
simulateMousePos = [-1, -1, -1]
selectChanged = False
# selectPlace: list[int] = [-1, 0, -1, -1, -1, -1]#type: type(0-8 check pos region), 0-rectange, 1-circle ; lu/centerx ; lb/centery ; ru/rad ; rb/- ; angle/-

counts = defaultdict(int)
object_str = ""
# index = 0
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

def getFrame():
    if CameraType == "basler":
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
        ret =  grabResult.GrabSucceeded()
        if not ret:
            print("lost connection to basler")
    else:
        ret, frame = cap.read()
        if not ret:
            print("no camera connected")
    return ret, frame

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
        (h ,w, _) = frame.shape
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
                    new_shape = [mark + i + 1, 0, int(new_center_x + 0.5), int(new_center_y + 0.5), radius, -1]
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
                _center, radius = defineCircle.define_circle_by_center_and_point(frame)
                if(_center != None):
                    selectPlace[2:5] = [_center[0], _center[1], radius]
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

fristFrame = None
selectSceneMask = None
selectMask = None
if len(sceneInfo) == 0:
    ret, fristFrame = getFrame()
    if not ret:
        quit()
    selectSceneMask = np.zeros_like(fristFrame)
    selectMask = np.zeros_like(fristFrame)

    if ret:
        sceneCenter, sceneRadius, sceneAngle = defineCircle.define_circle_by_three_points(fristFrame)
        sceneInfo = [sceneCenter[0], sceneCenter[1], sceneRadius, sceneAngle]
        selectSceneMask = cv2.circle(selectSceneMask, sceneCenter, sceneRadius, (0, 255, 0), 2)
        selectSceneMask = CircleSelect.draw_arrow(selectSceneMask, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
    else:
        quit()
else:
    ret, fristFrame = getFrame()
    if not ret:
        quit()
    selectSceneMask = np.zeros_like(fristFrame)
    selectMask = np.zeros_like(fristFrame)

    sceneCenter = (int(sceneInfo[0]), int(sceneInfo[1]))
    sceneRadius = int(sceneInfo[2])
    sceneAngle = sceneInfo[3]
    selectSceneMask = cv2.circle(selectSceneMask, sceneCenter, sceneRadius, (0, 255, 0), 2)
    selectSceneMask = CircleSelect.draw_arrow(selectSceneMask, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
drawSelectArea(selectMask, selectAreas)

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

while CameraType != "basler" or camera.IsGrabbing():
    ret, frame = getFrame()
    if not ret:
        break
    rectedFrame = copy.deepcopy(frame)

    if recordPredictResult:
        outRaw.write(frame)
        outWritten:bool = False

    if realMouseCenter[0] > 0:
        Predictframe = ProcessMouseNearRegion(realMouseCenter, frame)
    else:
        Predictframe = copy.deepcopy(frame) * availableMask

    if startTime < 0:
        startTime = time.time()
    if not ret:
        break

    receiveUnityTimeSuccess = -1
    lastReceiveUnityTime = -1
    onlineNumber = UnityShm.CheckOnlineClientsCount()
    if onlineNumber > 0 and UnityShm.care != "" :
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
                                            time.sleep(0.1)
                                        sync = True
                                        syncInd = 0
                                        UnityShm.WriteClear()
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
                    time.sleep(0.02)
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

    if frame_count % frame_rate_divider == 0:
        # if UnityShmPrepared:
        if UnityShmPrepared:
            results = model(Predictframe, verbose=False, conf = 0.7)
        else:
            results = []
        syncInd += 1 if syncInd >= 0 else 0
        if simulate:
            simulateMousePos[2] = syncInd if syncInd >= 0 else -1
            # print(simulateMousePos)
            UnityShm.WriteContent("pos:" + ";".join([str(i) for i in simulateMousePos]))
        # index = index +1
        for mask in [selectSceneMask, selectMask]:
        # 创建一个布尔掩码，标记 mask 中非黑色的区域
            non_black_mask = (mask != [0, 0, 0]).any(axis=-1)
        # 将 mask 的非黑色区域覆盖到 frame 上
            rectedFrame[non_black_mask] = mask[non_black_mask]

        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                # if class_id == 0:
                xyxy = np.array(box.xyxy[0].tolist(), int)
                realMouseCenter = [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
                rectedFrame = cv2.circle(rectedFrame, realMouseCenter, 5, (255,255,0), 10)
                if not simulate:
                    # UnityShm.WriteContent("pos" + ";".join([str(i) for i in xyxy]), True)
                    # UnityShm.WriteClear()
                    temp = realMouseCenter.copy()
                    temp.append(syncInd if syncInd >= 0 else -1)
                    simulateMousePos[2] = syncInd if syncInd >= 0 else -1
                    UnityShm.WriteContent("pos:" + ";".join([str(i) for i in temp]))

                break
            else:
                realMouseCenter = [-1, -1]


    # object_str = object_str +". " + key
    # for class_id, count in counts.items():
    #     object_str = object_str +f"{count} {class_id},"
    #     counts = defaultdict(int)
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

    if frame_count % 60 == 0:
        costTime = time.time() - startTime
        # print(str(60/costTime)+"fps")
        startTime = time.time()
    frame_count += 1

# object_str= object_str.strip(',').strip('.')
# print("reuslt:", object_str)
if selectChanged and PyWinMessageBox.YesOrNo("save current Info?", "save & load") == 'YES':
    f_selectionSaveTxt.seek(0, 0)
    f_selectionSaveTxt.truncate(0)
    f_selectionSaveTxt.write("scene:"+";".join([str(i) for i in sceneInfo]) + "\n")
    for selectArea in selectAreas:
        f_selectionSaveTxt.write("selectAreas:"+";".join([str(i) for i in selectArea]) + "\n")

f_selectionSaveTxt.close()

if camera != None:
    camera.Close()
if recordPredictResult:
    out.release()
    outRaw.release()
cv2.destroyAllWindows()
del UnityShm