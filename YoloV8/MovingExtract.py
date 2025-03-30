import os
import cv2
import time
import random
from ultralytics import YOLO
from math import*
import numpy as np
import keyboard

KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False)

MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

bg_subtractor=MOG2_subtractor

mediaName = "missedFrames"
mediaNamePure = mediaName.split('.')[0:-1][0] if mediaName.endswith('.mp4') else mediaName
useModel = False
conf = 0.4
sample = 0.2
minArea = 800
maxArea = 2400
if useModel:
    model = YOLO("models/TopViewMiniscopeBodyBest.pt")
picList = []
if(mediaName.endswith('.mp4')):
    camera = cv2.VideoCapture(mediaName)
else:
    camera = None
    if os.path.exists(mediaName):
        for file in os.listdir(mediaName):
            if random.randrange(0, 10, 1) <=(sample * 10) and (file.endswith('.jpg') or file.endswith('.png')):
                picList.append(os.path.join(mediaName, file))
    else:
        print("folder not exist")
        quit()

waitMillSec = 0

show = False

recFrame = 0
recDivider = 1
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

def getFrame() -> tuple[bool, np.ndarray, str]:
    if camera is None and len(picList) > 0:
        fname = picList.pop(0)
        pic = cv2.imread(fname)
        
        return True, pic, fname
    elif camera is not None:
        return camera.read(), ""
    else:
        return False, None, ""

_, fristFrame, _ = getFrame()

[height, width, _] = fristFrame.shape
availableMask = np.ones(fristFrame.shape, dtype= np.uint8)

if os.path.exists("tempMask.jpg"):
    tempMask = cv2.imread("tempMask.jpg")
else:
    tempMask = None
while True:
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
    ret, frame, fname = getFrame()
    if not ret:
        break

    frame[availableMask == 0] = 255
    # frame = frame * availableMask

    # pixel_sum = np.sum(frame, axis=2)
    # BlackMask = pixel_sum > tailColor
    # frame[mask] = preFrame[mask]
    # preFrame = frame.copy()
    foreground_mask = bg_subtractor.apply(frame)
    modelPredictfail = False
    
    if useModel:
        # foreground_mask = bg_subtractor.apply(frame)

        # # 如果大于240像素，则阈值设为255，如果小于则设为0    # 创建二值图像，它只包含白色和黑色像素
        # ret , threshold = cv2.threshold(foreground_mask.copy(), 150, 255, cv2.THRESH_BINARY)

        # # 膨胀扩展或加厚图像中的兴趣区域。
        # # threshold[BlackMask] = 0
        # threshold = cv2.medianBlur(threshold, 7)

        results = model(frame, verbose=False, conf = conf)
        
        for result in results:
            if not len(result.boxes):
                print(f"No object detected in file {fname}")
                modelPredictfail = True
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                # if class_id == 0:
                xyxy = np.array(box.xyxy[0].tolist(), int)

                [x,y,w,h] = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                # tempROI = np.sum(frame[y:y+h, x:x+w], axis= 2) / 3
                # tempResult = np.sum(cv2.threshold(tempROI, 200, 1, cv2.THRESH_BINARY)[1])
                # if recFrame % recDivider == 0:
                #     fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                #     cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y:y+h, x:x+w])
                #     fileName = mediaNamePure +str(int(recFrame / recDivider))
                #     cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                #     with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                #         file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
                #     print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                
                # if show:
                #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                # recFrame+=1

                mouseBlock = np.uint8(np.sum(frame[y:y+h, x:x+w, :], axis=2) / 3)
                _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.45, 255, cv2.THRESH_BINARY_INV)
                # _, tempResult = cv2.threshold(tempResult, np.max(mouseBlock) * 0.25, 255, cv2.THRESH_BINARY)
                tempResult = cv2.medianBlur(tempResult, 7) 
                # tempResultMixed = tempResult* np.right_shift(threshold[y:y+h, x:x+w], 7)
                tempResultMixed = cv2.dilate(tempResult, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 4)
                # tempContours, tempHier = cv2.findContours(tempResultMixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                tempContours, tempHier = cv2.findContours(tempResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # if show:
                #     cv2.imshow("tempChildPic",tempResult)
                #     # cv2.imshow("predicResult", cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2))
                #     # cv2.imshow("tempThreshold", threshold[y:y+h, x:x+w])
                #     cv2.imshow("tempMixedResult",tempResultMixed)
                for tempContour in tempContours:
                    tempChildArea = cv2.contourArea(tempContour)
                    if tempChildArea > minArea and tempChildArea < maxArea:
                        # print(tempChildArea)
                        (_x,_y,_w,_h) = cv2.boundingRect(tempContour)
                        if recFrame % recDivider == 0 and 2.5 > _w/_h > 0.4: 
                            fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                            cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y+_y:y+_y+_h, x+_x:x+_x+_w])
                            fileName = mediaNamePure +str(int(recFrame / recDivider))
                            cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                            with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                                file.write("0 "+ " ".join([str(i) for i in [(x+_x + _w*0.5)/width, (y+_y + _h*0.5)/height, _w/width, _h/height]]))
                            print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                            if show:
                                cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,0,255), 2)
                                cv2.putText(frame, str(tempChildArea), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            recFrame+=1
                            break
                    else:
                        if show:
                            cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,0,255), 2)
                            cv2.putText(frame, str(tempChildArea), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        # cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpeg", frame)
                        print(f"area not fit: {tempChildArea}")
                        
    
    if not useModel or modelPredictfail:

        # 每一帧既用于计算前景掩码，也用于更新背景。
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
            if tempArea > minArea*0.6 and tempArea < maxArea*1.5:
                (x,y,w,h) = cv2.boundingRect(contour)
                if show:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 2)
                print(f"tempArea: {tempArea}")
                mouseBlock = np.uint8(np.sum(frame[y:y+h, x:x+w, :], axis=2) / 3)
                _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.3, 255, cv2.THRESH_BINARY_INV)
                # _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.45, 255, cv2.THRESH_TOZERO_INV)
                # _, tempResult = cv2.threshold(tempResult, np.max(mouseBlock) * 0.25, 255, cv2.THRESH_BINARY)
                tempResult = cv2.medianBlur(tempResult, 7) 
                tempResultMixed = tempResult* np.right_shift(threshold[y:y+h, x:x+w], 7)
                tempResultMixed = cv2.dilate(tempResultMixed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
                tempContours, tempHier = cv2.findContours(tempResultMixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if show:
                    cv2.imshow("tempChildPic",tempResult)
                    cv2.imshow("tempThreshold", threshold[y:y+h, x:x+w])
                    cv2.imshow("tempMixedResult",tempResultMixed)
                for tempContour in tempContours:
                    tempChildArea = cv2.contourArea(tempContour)
                    if tempChildArea > minArea and tempChildArea < maxArea:
                        print(tempChildArea)
                        (_x,_y,_w,_h) = cv2.boundingRect(tempContour)
                        if recFrame % recDivider == 0 and 2.5 > _w/_h > 0.4: 
                            fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                            cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y+_y:y+_y+_h, x+_x:x+_x+_w])
                            fileName = mediaNamePure +str(int(recFrame / recDivider))
                            cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                            with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                                file.write("0 "+ " ".join([str(i) for i in [(x+_x + _w*0.5)/width, (y+_y + _h*0.5)/height, _w/width, _h/height]]))
                            print("Marked " + str(int(recFrame / recDivider)+1) + "Frames" + f"{"Detected" if modelPredictfail else ""}")
                        
                        if show:
                            cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,0,255), 2)
                        recFrame+=1
                # tempROI = np.sum(frame[y:y+h, x:x+w], axis= 2) / 3
                # tempResult = np.sum(cv2.threshold(tempROI, 200, 1, cv2.THRESH_BINARY)[1])
                # if tempResult > tempArea * 0.4:
                #     continue

                # if recFrame % recDivider == 0:
                #     fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                #     cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y:y+h, x:x+w])
                #     fileName = mediaNamePure +str(int(recFrame / recDivider))
                #     cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                #     with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                #         file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
                #     print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                
                # if show:
                #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                # recFrame+=1

        else:
            print(f"no movement dectected in frame{frameInd}")
        

    frameInd+=1
    
    if show:
        if not useModel:
            # cv2.imshow("Subtractor", foreground_mask)
            cv2.imshow("threshold", threshold)
        else:
            cv2.imshow("detection", frame)
        key = cv2.waitKey(waitMillSec) & 0xff
        # print(key)
        if key != 255:
            if key == 27:
                break
            elif key == 32:
                cv2.waitKey()

if camera is not None:
    camera.release()
cv2.destroyAllWindows()