import os
import cv2
import time
from ultralytics import YOLO
from math import*
import numpy as np
import keyboard

KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False)

MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

bg_subtractor=MOG2_subtractor

mediaName = "behavCam5.mp4"
mediaNamePure = mediaName.split('.')[0:-1][0]
camera = cv2.VideoCapture(mediaName)
waitMillSec = 1
show = True
continueShoot = True

recCount = 0
timeStr = time.strftime("%m%d%H%M%S", time.gmtime())
tempPicFolderName = "PoseExtract" + timeStr
# tempTxtFolderName = "OutputMouseBodyTxt" + timeStr
tempTxtFolderName = tempPicFolderName
tempROIFolderName = tempPicFolderName
if not os.path.exists(tempPicFolderName):
	os.makedirs(tempPicFolderName)
if not os.path.exists(tempTxtFolderName):
	os.makedirs(tempTxtFolderName)
# tailColor = 200

_, fristFrame = camera.read()
[height, width, _] = fristFrame.shape
availableMask = np.ones(fristFrame.shape, dtype= np.uint8)

model = YOLO("zj8nbest.pt")

# PosKeyPointsCount = 9
# PosKeyPointsName = ["鼻尖", "左耳", "右耳", "左前脚", "右前脚", "左后脚", "右后脚", "尾根", "尾中" ]
PosKeyPointsName = ["鼻尖", "左耳", "右耳", "尾根"]
PosKeyPointsCount = len(PosKeyPointsName)
clickColorLs:list[list[int]] = []
for i in range(1, PosKeyPointsCount + 1):
    tempcolor = i * (255 // PosKeyPointsCount)
    clickColorLs.append([tempcolor, 255 - abs(255 - int(tempcolor * 2)), 255 - tempcolor])

# def ClickEvent(event, x, y, flags, posLs:list[list[int]], colorMask:np.ndarray):
def ClickEvent(event, x, y, flags, param):
    # if cv2.getWindowProperty("poseMark") != -1:
    if posLs != None and len(posLs) < PosKeyPointsCount:
        if event == cv2.EVENT_LBUTTONDOWN:#可见点
            posLs.append([x, y, 2])
            # cv2.circle(colorMask, (x, y), 2, clickColorLs[len(posLs)], 4)
            # colorMask
        elif event == cv2.EVENT_RBUTTONDOWN:#不可见点
            posLs.append([x, y, 1])
            # cv2.circle(colorMask, (x, y), 2, clickColorLs[len(posLs)], 1)
        # elif event == -1:
        #     posLs.pop()
        cv2.waitKey(1)

def DrawPosPoints(rawFrame:np.ndarray, x, y, w, h):
    global posLs
    global recCount
    global width
    global height
    global continueShoot
    posLs = []
    # tempMaskLs:list[np.ndarray] = [] 
    scaled = 4
    img = rawFrame[y:y+ h, x:x+w]
    resizedImg = cv2.resize(img, (img.shape[1] * scaled, img.shape[0] * scaled), interpolation=cv2.INTER_LINEAR)
    tempMask = np.zeros(resizedImg.shape)

    enableDel:bool = True
    
    cv2.imshow("poseMark", img)
    cv2.setMouseCallback("poseMark", ClickEvent)
    while(True):
        cv2.waitKey(1)

        if not keyboard.is_pressed('backspace'):
            enableDel = True
        if keyboard.is_pressed('shift+n') and continueShoot == True:
            continueShoot = False
            print("stop shooting")

        if len(posLs):
            tempMask = np.zeros(resizedImg.shape)
            for i in range(len(posLs)):
                nowRadius = scaled * 2 if posLs[i][2] == 1 else scaled
                nowThick = 2  if posLs[i][2] == 1 else scaled * 2
                cv2.circle(tempMask, posLs[i][0:2], nowRadius, clickColorLs[i], nowThick)
        bool_mask = (tempMask > 0).astype(bool)
        tempImg = resizedImg.copy()
        tempImg[bool_mask] = tempMask[bool_mask]
        cv2.imshow("poseMark", tempImg)
        if keyboard.is_pressed('backspace') and enableDel:
            enableDel = False
            if len(posLs):
                posLs.pop()
                # print(f"delet one dot, now:{len(posLs)}")
                tempMask = np.zeros(resizedImg.shape)
                for i in range(len(posLs)):

                    cv2.circle(tempMask, posLs[i][0:2], 2, clickColorLs[i], posLs[i][2]*2)
            bool_mask = (tempMask > 0).astype(bool)
            tempImg = resizedImg.copy()
            tempImg[bool_mask] = tempMask[bool_mask]
            cv2.imshow("poseMark", tempImg)
            # tempMask = np.zeros(img.shape) if len(tempMaskLs) == 0 else tempMaskLs[-1]
        elif keyboard.is_pressed('enter'):
            if len(posLs) == PosKeyPointsCount:
                tempPointsStrLs:list[str] = []
                for points in posLs:
                    _x:float = float(points[0]/scaled+x)/w
                    _y:float = float(points[1]/scaled+y)/h
                    tempPointsStrLs.append(" ".join([str(i) for i in [_x, _y, points[2]]]))
                fileName = mediaNamePure + str(int(recCount)) + "pose"
                cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", tempImg)
                fileName = mediaNamePure +str(int(recCount))
                cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", rawFrame)
                with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                    file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]) + " " + " ".join(tempPointsStrLs))
                recCount += 1
                print(recCount)

                cv2.destroyWindow("poseMark")
                break
            elif len(posLs) == 0:
                 #跳过
                cv2.destroyWindow("poseMark")
                break

while True:
    tempMask = cv2.imread("tempMask.jpg")
    if(type(tempMask) != type(None) and tempMask.shape == fristFrame.shape):
        availableMask = tempMask
        
    gROI = cv2.selectROI("ROI frame", fristFrame * availableMask, False)

    availableMask[gROI[1]:(gROI[1] + gROI[3]), gROI[0]:(gROI[0] + gROI[2])] = 0

    if keyboard.is_pressed('s'):
        cv2.imwrite("tempMask.jpg", availableMask)

    if gROI == (0,0,0,0):
        cv2.destroyWindow("ROI frame")
        break

while True:
    ret, frame = camera.read()
    frame = frame * availableMask

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            xyxy = np.array(box.xyxy[0].tolist(), int)
            center = [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
            x = xyxy[0]
            y = xyxy[1]
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]

            if keyboard.is_pressed('n') and continueShoot == False:
                print("continue shooting")
                continueShoot = True

            
            if continueShoot:
                DrawPosPoints(frame, x, y, w, h)
            if show:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
            # print(tempArea)
            
            break

    if show:
        cv2.imshow("detection", frame)
        key = cv2.waitKey(waitMillSec) & 0xff
        # print(key)
        if key != 255:
            if key == 27:
                break
            elif key == 32:
                cv2.waitKey()
        # elif key == 38:
        #     tailColor += 2
        #     print(tailColor)
        # elif key == 40:
        #     tailColor -= 2
        #     print(tailColor)
        # elif key == 0:
        #     waitMillSec = 2
        #     pass
        # else:
        #     waitMillSec = 17
        # print("key press")

camera.release()
cv2.destroyAllWindows()