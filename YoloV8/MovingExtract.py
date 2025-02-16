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

mediaName = "01_17_1842outputraw.mp4"
mediaNamePure = mediaName.split('.')[0:-1][0]
camera = cv2.VideoCapture(mediaName)
waitMillSec = 1
useModel = True
if useModel:
    model = YOLO("best.pt")

show = True

recFrame = 0
recDivider = 5
timeStr = time.strftime("%m%d%H%M%S", time.gmtime())
tempPicFolderName = "OutputMouseBodyPic" + timeStr
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

frameInd:int = 0

while True:
    ret, frame = camera.read()
    frame[availableMask == 0] = 255
    # frame = frame * availableMask

    # pixel_sum = np.sum(frame, axis=2)
    # BlackMask = pixel_sum > tailColor
    # frame[mask] = preFrame[mask]
    # preFrame = frame.copy()
    
    if useModel:
        results = model(frame, verbose=False, conf = 0.7)
        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                # if class_id == 0:
                xyxy = np.array(box.xyxy[0].tolist(), int)

                [x,y,w,h] = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                tempROI = np.sum(frame[y:y+h, x:x+w], axis= 2) / 3
                tempResult = np.sum(cv2.threshold(tempROI, 200, 1, cv2.THRESH_BINARY)[1])
                if recFrame % recDivider == 0:
                    fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                    cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y:y+h, x:x+w])
                    fileName = mediaNamePure +str(int(recFrame / recDivider))
                    cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                    with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                        file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
                    print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                
                if show:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                recFrame+=1
    
    else:

        # 每一帧既用于计算前景掩码，也用于更新背景。
        foreground_mask = bg_subtractor.apply(frame)

        # 如果大于240像素，则阈值设为255，如果小于则设为0    # 创建二值图像，它只包含白色和黑色像素
        ret , threshold = cv2.threshold(foreground_mask.copy(), 200, 255, cv2.THRESH_BINARY)

        # 膨胀扩展或加厚图像中的兴趣区域。
        # threshold[BlackMask] = 0
        threshold = cv2.medianBlur(threshold, 7)
        dilated = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 4)

        # 查找轮廓
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查每个轮廓是否超过某个值，如果超过则绘制边界框
        for contour in contours:
            tempArea = cv2.contourArea(contour)
            if tempArea > 1200 and tempArea < 6000:
                (x,y,w,h) = cv2.boundingRect(contour)
                # if show:
                #     cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 2)
                # print(f"tempArea: {tempArea}")
                # mouseBlock = np.uint8(np.sum(frame[y:y+h, x:x+w, :], axis=2) / 3)
                # _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.45, 255, cv2.THRESH_TOZERO_INV)
                # _, tempResult = cv2.threshold(tempResult, np.max(mouseBlock) * 0.25, 255, cv2.THRESH_BINARY)
                # tempResult = cv2.medianBlur(tempResult, 7) 
                # tempResultMixed = tempResult* np.right_shift(threshold[y:y+h, x:x+w], 7)
                # tempResultMixed = cv2.dilate(tempResultMixed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 4)
                # tempContours, tempHier = cv2.findContours(tempResultMixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # if show:
                #     cv2.imshow("tempChildPic",tempResult)
                #     cv2.imshow("tempThreshold", threshold[y:y+h, x:x+w])
                #     cv2.imshow("tempMixedResult",tempResultMixed)
                # for tempContour in tempContours:
                #     tempChildArea = cv2.contourArea(tempContour)
                #     if tempChildArea > 2200 and tempChildArea < 5500:
                #         # print(tempChildArea)
                #         (_x,_y,_w,_h) = cv2.boundingRect(tempContour)
                        # if recFrame % 10 == 0 and 2.5 > _w/_h > 0.4: 
                        # fileName = mediaNamePure + str(int(recFrame / 10)) + "ROI"
                        # cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y+_y:y+_y+_h, x+_x:x+_x+_w])
                        # fileName = mediaNamePure +str(int(recFrame / 10))
                        # cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                        # with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                        #     file.write("0 "+ " ".join([str(i) for i in [(x+_x + _w*0.5)/width, (y+_y + _h*0.5)/height, _w/width, _h/height]]))
                        # print("Marked " + str(int(recFrame / 10)+1) + "Frames")
                        
                        # if show:
                        #     cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,0,255), 2)
                        # recFrame+=1
                tempROI = np.sum(frame[y:y+h, x:x+w], axis= 2) / 3
                tempResult = np.sum(cv2.threshold(tempROI, 200, 1, cv2.THRESH_BINARY)[1])
                if tempResult > tempArea * 0.4:
                    continue

                if recFrame % recDivider == 0:
                    fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                    cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y:y+h, x:x+w])
                    fileName = mediaNamePure +str(int(recFrame / recDivider))
                    cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                    with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                        file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
                    print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                
                if show:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                recFrame+=1
                # if show:
                #     cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
                # print(tempArea)
                
                break

        else:
            print(f"no movement dectected in frame{frameInd}")
        

    frameInd+=1
    
    if show and not useModel:
        cv2.imshow("Subtractor", foreground_mask)
        cv2.imshow("threshold", threshold)
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
    
    # elif key == 120 or key == 88:
    #     gROI = cv2.selectROI("ROI frame", frame, False)
    #     ROI = frame[gROI[1]:(gROI[1] + gROI[3]), gROI[0]:(gROI[0] + gROI[2])]

    #     cv2.waitKey()
camera.release()
cv2.destroyAllWindows()