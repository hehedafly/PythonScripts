import os 
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import math
import time
import shutil
import bisect
import threading
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation

import tkinter as tk
from tkinter import filedialog
from math import*
from ParseEvent import ParseEvent


def Distance(pos1:list, pos2:list) -> float:
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

root = tk.Tk()
root.withdraw()

mouseNames:list[str] = []
LogsFolders:list[str] = []
configs = []

All_folder_paths = "D:/Unity/PythonFiles/Logs/tempTest"
if All_folder_paths == "":
    All_folder_paths = filedialog.askdirectory(initialdir = os.getcwd(), title="选择包含指定批次Log数据的文件夹")
if not All_folder_paths:
    exit()

# 遍历每个选择的文件夹
for folder_path in os.listdir(All_folder_paths):
    if not folder_path.startswith("#") and os.path.isdir(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(folder_path)
        LogsFolders.append(os.path.join(All_folder_paths, folder_path))
    elif os.path.isfile(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(All_folder_paths)
        LogsFolders.append(All_folder_paths)
        break


recfileList = []
recfileListTimeList = []
posfileList = []
logeventList = []
logeventRenamedList = []
logeventTimeList = []


def GetPklFileName(_type:str, _mouseName, _day):
    if _type not in ["rec", "pos", "evt"]:
        raise ValueError("Invalid type")

    if _type == "rec":
        return f"rec_raw_res_{_mouseName}_{_day}.txt"
    elif _type == "pos":
        return f"pos_raw_res_{_mouseName}_{_day}.txt"
    elif _type == "evt":
        return f"evt_raw_{_mouseName}_{_day}.txt"
            
    
    pass

limitedColumns = ['type', 'delta time', 'mode', 'trial', 'lickPos', 'result']
for mouseIndex in LogsFolders:
    recfileList.append([])
    posfileList.append([])
    logeventList.append([])
    logeventRenamedList.append([])
    recfileListTimeList.append([])
    logeventTimeList.append([])

    for file in os.listdir(mouseIndex):
        if file.endswith("rec.txt"):    
            recfileList[-1].append(file)
            recfileListTimeList[-1].append(time.strftime("%m_%d", time.localtime(os.path.getmtime(os.path.join(mouseIndex, file)))))
        elif file.endswith("pos.txt"):
            posfileList[-1].append(file)
        elif file.endswith(".evt"):
            if "_evt" not in file:
                logeventList[-1].append(mouseIndex + "/" + file)
                logeventRenamedList[-1].append("")

            else:
                logeventList[-1].append("")
                logeventRenamedList[-1].append(mouseIndex + "/" + file)
            logeventTimeList[-1].append(time.strftime("%m_%d", time.localtime(os.path.getmtime(os.path.join(mouseIndex, file)))))
    recfileList[-1].sort()
    posfileList[-1].sort()

def readToDataRows(_fname, _dataRows, _addtion:list = None):
    _config = ""
    with open(_fname, 'r') as file:
        for line in file:
            line.strip()
            if IsValidDataLine(line):
                # 按制表符分割行，并添加到数据行列表中
                _dataRows.append(line.strip().split('\t')[0:len(limitedColumns)])
                #if len(dataRows) > 1:
                    #dataRows[-1][1] = float(dataRows[-1][1])
                #dataRowsTyped.append(dataRows[-1])
            elif line.startswith("{"):
                _config += line + "\n"
            if _addtion is not None:
                if "Trial initialized" in line:
                    _addtion.clear()
                if "started at " in line:
                    _addtion.append(line[line.index("started at "):line.index("started at ") + 1])

    return _config

sceneinfo = None
selectedArea = '''34	0	378	478	120	1
33	0	477	745	120	1
32	0	736	864	120	1
36	0	764	120	120	1
35	0	497	219	120	1
37	0	1023	239	120	1
39	0	1003	765	120	1
38	0	1122	506	120	1'''
selectedArea = selectedArea.strip().split('\n')


def IsValidPosLine(tempDfValue) -> bool:
    if len(tempDfValue) == 6:
        try:
            _y = int(tempDfValue['y'])
            int(tempDfValue['frameInd'])
            _t = float(tempDfValue['TimeInUnitySecFromTrialStart'])
        
            if _y in [0, 1] and int(_t) in [0, 1]:
                return False
            else:
                return True
        except:
            return False
    return False

def IsValidDataLine(line:str) -> bool:
    splited = line.split('\t')
    if len(splited) == 6:
        try:
            float(splited[1])
            int(splited[3])
            return True
        except:
            return False
    return False

for mouseIndex in range(0, len(recfileList)):#fileList按mouse排序
    for i, DailyRecord in enumerate(recfileList[mouseIndex]):
        _datarows = []
        _config = readToDataRows(LogsFolders[mouseIndex] + "/" + DailyRecord, _datarows)
        # configs[-1].append(_config)

        rec_raw_res_file = LogsFolders[mouseIndex] + "/" + f"rec_raw_res_{DailyRecord}.pkl"
        if not os.path.exists(rec_raw_res_file):
            nowDf = pd.DataFrame(_datarows, columns=limitedColumns)
            nowDf = nowDf.drop(nowDf.index[nowDf['type'] == "type"])
            nowDf['delta time'] = np.float64(nowDf['delta time'])
            nowDf['config'] = [""]*len(nowDf)
            nowDf['config'].iloc[0] = _config
            nowDf.to_pickle(rec_raw_res_file)
        # else:
        #     nowDf = pd.read_pickle(rec_raw_res_file)


        # AllMouseRecMainDfs[-1].append(nowDf)
        recfileName = recfileList[mouseIndex][i].split("_rec")[0]
        evtfilename = LogsFolders[mouseIndex] + "/" + "evt_raw_" +recfileName + ".evt"
        if evtfilename not in logeventRenamedList[mouseIndex]:
            date = recfileListTimeList[mouseIndex][i]
            if date in logeventTimeList[mouseIndex]:
                index = logeventTimeList[mouseIndex].index(date)
                if logeventRenamedList[mouseIndex][index].split("/")[-1].split("_evt.evt")[0] != DailyRecord.split('_rec')[0]:
                    new_evt_name = evtfilename
                    shutil.copy2(logeventRenamedList[mouseIndex][index], new_evt_name)

    for DailyPos in posfileList[mouseIndex]:
        _datarows = []
        sceneinfo = []
        tempAreaLs = []
        pos_raw_res_file = LogsFolders[mouseIndex] + "/" + f"pos_raw_res_{DailyPos}.pkl"

        if not os.path.exists(pos_raw_res_file):
            nowDf = pd.read_csv(LogsFolders[mouseIndex] + "/" + DailyPos,sep='\t',names=['x','y', 'syncInd', "100*pythonTime", "frameInd", "TimeInUnitySecFromTrialStart"], dtype=str)
            while not IsValidPosLine(nowDf.iloc[0].astype(str)):
                if nowDf.iloc[0].isnull().sum() == 2:
                    sceneinfo = nowDf.iloc[0].astype(float, errors='ignore').values
                    if type(sceneinfo[0]) == str:
                        sceneinfo = []
                else:
                    try:
                        tempAreaLs = tuple(nowDf.iloc[0].astype(int).values)
                        # if tempAreaLs not in selectedArea:
                        #     selectedArea.append(tempAreaLs)
                    except:
                        pass
                nowDf = nowDf.iloc[1:]

            nowDf = nowDf.drop_duplicates(subset='syncInd', keep='first')
            nowDf.loc[:,'x':'frameInd'] = nowDf.loc[:,'x':'frameInd'].astype(int)
            nowDf['TimeInUnitySecFromTrialStart'] = nowDf['TimeInUnitySecFromTrialStart'].astype(float)
            nowDf['sceneinfo'] = ['']* len(nowDf)
            nowDf['sceneinfo'].iloc[0] = str(sceneinfo)
            nowDf['sceneinfo'].iloc[1] = str(tempAreaLs)
            nowDf.index = np.array(range(0, len(nowDf)))
            nowDf.to_pickle(pos_raw_res_file)
        # else:
        #     nowDf = pd.read_pickle(pos_raw_res_file)
            
print(f"{len(recfileList)} mouse added")


for folder_path in os.listdir(All_folder_paths):
    if not folder_path.startswith("#") and os.path.isdir(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(folder_path)
        LogsFolders.append(os.path.join(All_folder_paths, folder_path))
    elif os.path.isfile(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(All_folder_paths)
        LogsFolders.append(All_folder_paths)
        break


recrawfileList = []
posrawfileList = []
logeventList = []

limitedColumns = ['type', 'delta time', 'mode', 'trial', 'lickPos', 'result']
for mouseIndex in LogsFolders:
    recrawfileList.append([])
    posrawfileList.append([])
    logeventList.append([])

    for file in os.listdir(mouseIndex):
        if file.startswith("rec_raw_res_"):    
            recrawfileList[-1].append(file)
        elif file.startswith("pos_raw_res_"):
            posrawfileList[-1].append(file)
        elif file.startswith("evt_raw_") and file.endswith(".evt"):
            logeventList[-1].append(file)

ignoreFristTrials = 15
ignoreLastTrials = 15
maxTrials = ignoreFristTrials + 320 + ignoreLastTrials

global trialGroupCount, trialPerGroup
trialGroupCount= -1
trialPerGroup = 2 if trialGroupCount < 0 else -1

class Day:
    def __init__(self, _RecMainDf, _PosMainDf, _name, _day, _eventFile = '', _savePath = ''):
        self.trials                  :list[pd.DataFrame]= []
        self.trialIndexes            :list= []
        self.startIndices            :list= []
        self.EndIndices              :list= []
        self.licks                   :list= []
        self.licksPos                :list= []
        self.lickInterval            :list= []
        self.lickIntervalLog         :list= []
        self.trialResults            :list= []
        self.trialElapsedTime        :list= []
        self.trialIntervalTime       :list= []
        self.trialAccuracyInGroup    :list= []
        self.trialElapsedTimeInGroup :list= []
        self.trialIntervalInGroup    :list= []
        self.trialfhDistanceInGroup  :list= []
        self.trialfuDistanceInGroup  :list= []
        self.trialfhElaspedTime      :list= []
        self.trialfhElaspedTimeInGroup:list= []
        
        self.posPerTrial             :list= []
        self.uposPerTrial            :list= []
        self.posSegPerTrial          :list= []
        self.speedPerTrial           :list= []
        self.fuDistancePerTrial      :list= []
        self.fhDistancePerTrial      :list= []
        self.uDistancePerTrial       :list= []
        self.DistancePerTrial        :list= []
        self.trialPosMark            :list= []
        self.trialTimeMark           :list= []
        self.trialTimeMarkCollected  :list= []
        # self.trackDirection          :list= []
        # self.posPolarPerTrial         :list= []

        self.RecMainDf:pd.DataFrame = pd.read_pickle(_RecMainDf)
        self.PosMainDf:pd.DataFrame = pd.read_pickle(_PosMainDf)
        # self.RecMainDfClear:pd.DataFrame = _RecMainDf
        self.PosMainDfClear:pd.DataFrame = None
        self.trialPerGroup = -1
        self.day = _day
        self.name = _name
        self.eventFile = _eventFile
        self.fullEventRecord = None
        self.savePath = _savePath if _savePath!= '' else os.path.join(os.getcwd(), 'tempSummaryLogs')

    def Process(self): 
        print("Processing mouse " + self.name + " day " + str(self.day))
        init = self.RecMainDf.index[self.RecMainDf['type'] == "init"]
        if (len(init)):
            init = max(init)
        else:
            init = 0
        self.RecMainDf = self.RecMainDf[init:]

        self.startIndices    =          self.RecMainDf.index[self.RecMainDf['type'] == "start"]#trial处理均以开始结束Index为标准
        self.EndIndices      =          self.RecMainDf.index[self.RecMainDf['type'] == "end"]
        self.licks           = np.array(self.RecMainDf.loc[self.RecMainDf["type"] == "lick"]["delta time"])
        self.startIndices.delete(range(len(self.EndIndices), len(self.startIndices)))

        self.PosMainx = []
        self.PosMainy = []
        self.SmoothTrack()  #add 'x_smooth' and 'y_smooth' columns to PosMainDf

        dt = np.gradient(self.PosMainDf['TimeInUnitySecFromTrialStart'])  # 使用梯度处理非均匀时间戳
        dx = np.gradient(self.PosMainDf['smooth_x'])
        dy = np.gradient(self.PosMainDf['smooth_y'])
        unitDistance = np.sqrt((dx)**2 + (dy)**2)
        speed = np.sqrt((dx/dt)**2 + (dy/dt)**2)
        self.PosMainDf['speed'] = speed
        self.PosMainDf['unitDistance'] = unitDistance
        self.PosMainDf['elaspedtime'] = self.PosMainDf['TimeInUnitySecFromTrialStart'] - self.PosMainDf['TimeInUnitySecFromTrialStart'].shift(1)
        self.PosMainDf.loc[0, 'elaspedtime'] = 0
        # time = self.PosMainDf.loc[(self.PosMainDf['elaspedtime'] > 0.5) | (self.PosMainDf['unitDistance'] > 40),'TimeInUnitySecFromTrialStart']
        time = self.PosMainDf.loc[self.PosMainDf['unitDistance'] > 40,'TimeInUnitySecFromTrialStart']

        self.PosMainDfClear = pd.DataFrame.copy(self.PosMainDf, deep=True)
        # DrawTrack(self.PosMainDf['x'].values, self.PosMainDf['y'].values, unitDistance, title = "orogin track")
        # plt.hist(unitDistance, bins=10, alpha=0.5, density=False, rwidth= 0.9, label='step distance')
        # DrawTrack(self.PosMainDf['smooth_x'].values, self.PosMainDf['smooth_y'].values, unitDistance, title = "smoothed track")        
        # popOffset = 0
        segment = []
        fristIgnoredTrials = 0
        lastIgnoredTrials = 0
        eventType = ['OGManipulate', 'miniscopeRecord']
        self.deviceEventRecord = {}
        for event in eventType:
            tempDf = self.RecMainDf[self.RecMainDf['type'] == event]
            dif = np.array(tempDf['lickPos'].values, dtype = int)
            if len(dif):
                dif = np.insert(dif[1:] - dif[:-1], 0, 1)
                dif = np.array(tempDf.index)[np.where(dif != 0)[0]]
            else:
                dif = tempDf.index
            self.deviceEventRecord[event] = tempDf.loc[dif]

        for i in range(1, min(maxTrials, len(self.EndIndices))):
            # _i = i - popOffset
            self.trials.append(         self.RecMainDf.loc[self.startIndices[i-1] : self.startIndices[i], :])
            tempResult:int      = int(  self.RecMainDf.loc[self.EndIndices[     i - 1]]["result"])
            tempStartTime       =       self.RecMainDf.loc[self.startIndices[   i - 1]]['delta time']
            tempTrackStartTime  =       self.RecMainDf.loc[self.EndIndices[     i - 2]]['delta time'] if i > 1 else self.RecMainDf.loc[self.startIndices[   i - 1]]['delta time']
            tempEndTime         =       self.RecMainDf.loc[self.EndIndices[     i - 1]]['delta time']
            
            tempDf = self.trials[-1]
            # if i % 50 == 0:
            #     print("Processing trial " + str(i) + " of " + str(len(self.startIndices)))

            if self.CheckTrialIntegrity(tempDf) and tempResult >= 0 and not time.between(tempStartTime, tempEndTime, inclusive='both').any() and tempEndTime - tempStartTime < 360:
                self.trialIndexes.append(       int(self.RecMainDf.loc[self.startIndices[i - 1]]['trial']))
                # self.trialIndexes.append(       i)
                self.licksPos.append(           self.RecMainDf.loc[self.startIndices[i - 1]]["lickPos"])
                self.trialResults.append(       self.RecMainDf.loc[self.EndIndices  [i - 1]]["result"])
                self.trialElapsedTime.append(   self.RecMainDf.loc[self.EndIndices  [i-1]]['delta time'] - self.RecMainDf.loc[self.startIndices[i-1]]['delta time'])
                self.trialIntervalTime.append(  self.RecMainDf.loc[self.startIndices[i]]['delta time'] - self.RecMainDf.loc[self.EndIndices[i-1]]['delta time'])
                self.posPerTrial.append(        self.PosMainDf.loc[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempTrackStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime), :])
                self.uposPerTrial.append(       self.PosMainDf.loc[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime), :])
                # self.posSegPerTrial.append(     self.SpeedSegment(self.posPerTrial[-1]['speed'], self.posPerTrial[-1]))
                # self.posSegPerTrial.append([])
                self.speedPerTrial.append(      speed[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempTrackStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]])
                self.fuDistancePerTrial.append( unitDistance[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempTrackStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]])
                self.uDistancePerTrial.append(  unitDistance[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]])
                self.DistancePerTrial.append(np.sum(self.uDistancePerTrial[-1]))

                tempStayTime = tempDf.loc[tempDf['type'] == 'stay']['delta time'].values[0]
                self.trialPosMark.append([self.GetPosOrIndAtTime(tempStayTime - 0.5), self.GetPosOrIndAtTime(tempStayTime - 0.25)])
                self.trialfhElaspedTime.append(tempStayTime - tempStartTime)
                self.fhDistancePerTrial.append(np.sum(unitDistance[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempStayTime)]]))
                self.trialTimeMark.append([self.GetPosOrIndAtTime(tempStartTime, self.posPerTrial[-1], needInd=True), self.GetPosOrIndAtTime(tempStayTime - 0.5, self.posPerTrial[-1], needInd=True), self.GetPosOrIndAtTime(tempStayTime, self.posPerTrial[-1], needInd=True)])
                self.trialTimeMarkCollected.append([tempStartTime, tempStayTime, tempEndTime])
                # dir, polar = self.TrackNormalization(np.asarray([self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values]).T, 20, 20)
                # self.trackDirection.append(tempDf.loc[tempDf['type'] == 'start']['lickPos'])
                # self.posPolarPerTrial.append(polar)
                # if self.day >=3:
                #     DrawTrack(self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values, self.unitDistancePerTrial[-1], addInfo=f'{self.name}_{self.day}_{np.sum(self.unitDistancePerTrial[-1])/sceneinfo[2]:.2f}_trial{i}', show=False, save=True, lw=0.6, pointSize=0.3)
                # DrawTrack(self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values, addInfo=f'Level{np.sum(self.unitDistancePerTrial[-1])//2500}_{self.name}_{self.day}_{i - 1}', show=False, save=True)
                
            else:
                posDfDropIndexes = self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]
                segment.append(posDfDropIndexes[0])
                self.trials.pop()
                self.PosMainDfClear = self.PosMainDfClear.drop(posDfDropIndexes)
                # popOffset += 1
                if i < ignoreFristTrials:
                    fristIgnoredTrials += 1
                elif i > len(self.EndIndices) - ignoreLastTrials:
                    lastIgnoredTrials += 1
                # tempDf = self.PosMainDf.loc[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempTrackStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime), :]
                # DrawTrack(tempDf['smooth_x'].values, tempDf['smooth_y'].values, unitDistance[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempTrackStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]], addInfo=f'abnormal_{self.name}_{self.day}_{i}', show=False, save=True)
            
        # DrawTrack(self.PosMainDfClear['smooth_x'].values, self.PosMainDfClear['smooth_y'].values, unitDistance[self.PosMainDfClear.index], segment=segment, title = "smoothed track", save=True, addInfo=f'{self.name}_{self.day}')        
        fullTrialCount = len(self.trials)
        _startInd = ignoreFristTrials - fristIgnoredTrials
        _endInd = min(fullTrialCount - ignoreLastTrials + lastIgnoredTrials, maxTrials)
        self.trials             = self.trials[              _startInd: _endInd]
        self.trialIndexes       = self.trialIndexes[        _startInd: _endInd]
        self.licksPos           = self.licksPos[            _startInd: _endInd]
        self.trialResults       = self.trialResults[        _startInd: _endInd]
        self.trialElapsedTime   = self.trialElapsedTime[    _startInd: _endInd]
        self.trialfhElaspedTime = self.trialfhElaspedTime[  _startInd: _endInd]
        self.trialIntervalTime  = self.trialIntervalTime[   _startInd: _endInd]
        self.posPerTrial        = self.posPerTrial[         _startInd: _endInd]
        self.uposPerTrial       = self.uposPerTrial[        _startInd: _endInd]
        self.posSegPerTrial     = self.posSegPerTrial[      _startInd: _endInd]
        self.speedPerTrial      = self.speedPerTrial[       _startInd: _endInd]
        self.fuDistancePerTrial = self.fuDistancePerTrial[  _startInd: _endInd]
        self.fhDistancePerTrial = self.fhDistancePerTrial[  _startInd: _endInd]
        self.uDistancePerTrial = self.uDistancePerTrial[    _startInd: _endInd]
        self.DistancePerTrial = self.DistancePerTrial[      _startInd: _endInd]
        self.trialPosMark          = self.trialPosMark[           _startInd: _endInd]
        self.trialTimeMark         = self.trialTimeMark[          _startInd: _endInd]
        self.trialTimeMarkCollected = self.trialTimeMarkCollected[_startInd: _endInd]
        # self.trackDirection        = self.trackDirection[         _startInd: _endInd]
        # self.posPolarPerTrial       = self.posPolarPerTrial[        _startInd: _endInd]
    
        self.trialResults = np.int8(self.trialResults)
        self.licksPos = np.int16(self.licksPos)
        self.trialTimeMarkCollected = np.array(self.trialTimeMarkCollected)

        self.lickInterval       = self.licks[1:] - self.licks[0: -1]
        self.lickInterval       = self.lickInterval[self.lickInterval > 0]
        self.lickIntervalLog    = np.log(1/self.lickInterval)

        SummaryDf = pd.DataFrame()
        SummaryDf['trials'] = self.trials
        SummaryDf['trialIndexes'] = self.trialIndexes
        SummaryDf['lickPos'] = self.licksPos
        SummaryDf['result'] = self.trialResults
        SummaryDf['trialElapsedTime'] = self.trialElapsedTime
        SummaryDf['trialfhElaspedTime'] = self.trialfhElaspedTime
        SummaryDf['trialIntervalTime'] = self.trialIntervalTime
        SummaryDf['posPerTrial'] = self.posPerTrial
        SummaryDf['uposPerTrial'] = self.uposPerTrial
        SummaryDf['speedPerTrial'] = self.speedPerTrial
        SummaryDf['fuDistancePerTrial'] = self.fuDistancePerTrial
        SummaryDf['fhDistancePerTrial'] = self.fhDistancePerTrial
        SummaryDf['uDistancePerTrial'] = self.uDistancePerTrial
        SummaryDf['DistancePerTrial'] = self.DistancePerTrial
        SummaryDf['trialPosMark'] = [str(mark) for mark in self.trialPosMark]
        SummaryDf['trialTimeMark'] = [str(mark) for mark in self.trialTimeMark]
        SummaryDf['trialTimeMarkCollected'] = [str(mark) for mark in self.trialTimeMarkCollected]

        SummaryDf[
            ['trialIndexes', 'lickPos', 'result', 'trialElapsedTime', 'trialfhElaspedTime', 'trialIntervalTime', 'fuDistancePerTrial', 'fhDistancePerTrial', 'uDistancePerTrial', 'DistancePerTrial', 'trialPosMark', 'trialTimeMark', 'trialTimeMarkCollected']
            ].to_csv(f'{self.savePath}/{self.name}_{self.day}_summary.csv', index=True, header=True)

        SummaryDf.to_pickle(f'{self.savePath}/{self.name}_{self.day}_summary.pkl')

        if(trialGroupCount > 0):
            self.trialPerGroup = max(1, ceil(len(self.trials)/trialGroupCount))
        else:
            self.trialPerGroup = trialPerGroup

        for i in range(0, ceil(len(self.trialResults) / self.trialPerGroup)):
            self.trialAccuracyInGroup.append(   np.sum(self.trialResults[           i * self.trialPerGroup:  min(len(self.trialResults)             , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialResults)      - i * self.trialPerGroup))
            self.trialElapsedTimeInGroup.append(np.sum(self.trialElapsedTime[       i * self.trialPerGroup:  min(len(self.trialElapsedTime)         , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialElapsedTime)  - i * self.trialPerGroup))
            self.trialfhElaspedTimeInGroup.append(np.sum(self.trialfhElaspedTime[   i * self.trialPerGroup:  min(len(self.trialfhElaspedTime)       , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialfhElaspedTime) - i * self.trialPerGroup))
            self.trialIntervalInGroup.append(   np.sum(self.trialIntervalTime[      i * self.trialPerGroup:  min(len(self.trialIntervalTime)        , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialIntervalTime) - i * self.trialPerGroup))
            self.trialfhDistanceInGroup.append( np.sum(self.fhDistancePerTrial[     i * self.trialPerGroup:  min(len(self.fhDistancePerTrial)       , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.fhDistancePerTrial)- i * self.trialPerGroup))
            self.trialfuDistanceInGroup.append( np.sum(self.trialfuDistanceInGroup[ i * self.trialPerGroup:  min(len(self.trialfuDistanceInGroup)   , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.fuDistancePerTrial)- i * self.trialPerGroup))

        self.trialElapsedTimeMean    = np.mean(self.trialElapsedTime)
        self.trialIntervalTimeMean   = np.mean(self.trialIntervalTime)

        # plt.hist(self.trialElapsedTime, range=(0, 120), bins=10, alpha=0.5, density=True, rwidth= 0.9, label='Trial Elapsed Time')
        if os.path.exists(self.eventFile):
            self.fullEventRecord = ParseEvent(self.eventFile)

        print("Day " + str(self.day) + " processed.")
        return True
    
    def Load(self) -> bool:
        if os.path.exists(f'{self.savePath}/{self.name}_{self.day}_summary.pkl'):
            try:
                tempLoadedDf:pd.DataFrame = pd.read_pickle(f'{self.savePath}/{self.name}_{self.day}_summary.pkl')
                _columns = tempLoadedDf.columns.tolist()
                self.trials = tempLoadedDf[_columns[:_columns.index('trialIndexes')]]
                self.trialIndexes = tempLoadedDf['trialIndexes'].values
                self.licksPos = tempLoadedDf['lickPos'].values
                self.trialResults = tempLoadedDf['result'].values
                self.trialElapsedTime = tempLoadedDf['trialElapsedTime'].values
                self.trialfhElaspedTime = tempLoadedDf['trialfhElaspedTime'].values
                self.trialIntervalTime = tempLoadedDf['trialIntervalTime'].values
                self.posPerTrial = tempLoadedDf['posPerTrial'].values
                self.uposPerTrial = tempLoadedDf['uposPerTrial'].values
                # self.posSegPerTrial = tempLoadedDf['posSegPerTrial'].values
                self.speedPerTrial = tempLoadedDf['speedPerTrial'].values
                self.fuDistancePerTrial = tempLoadedDf['fuDistancePerTrial'].values
                self.fhDistancePerTrial = tempLoadedDf['fhDistancePerTrial'].values
                self.uDistancePerTrial = tempLoadedDf['uDistancePerTrial'].values
                self.DistancePerTrial = tempLoadedDf['DistancePerTrial'].values
                self.trialPosMark = tempLoadedDf['trialPosMark'].values
                self.trialTimeMark = tempLoadedDf['trialTimeMark'].values
                self.trialTimeMarkCollected = tempLoadedDf['trialTimeMarkCollected'].values
            except Exception as e:
                print(f"Error loading summary file: {e}")
                return False
            return True
        else:
            return False

    def CheckTrialIntegrity(self, tempDf:pd.DataFrame) -> bool:
        essentialTypes = ['stay']
        for essentialType in essentialTypes:
            if essentialType not in tempDf['type'].values:
                return False
        return True

    # def GetPosOrIndAtTime(self, times, df:pd.DataFrame = None, needInd = False, tolerance = 0.02):
    def GetPosOrIndAtTime(self, times, df:pd.DataFrame = None, refTimes = None, needInd = False, tolerance = 0.02):

        # def GetPosOrIndAtTimeBatch(self, times_list, df: pd.DataFrame = None, needInd=False, tolerance=0.02):
        if df is None:
            df = self.PosMainDf
        # if refTimes is None:
        #     refTimes = 

        singleResult = False
        if type(times) is not list and type(times) is not np.ndarray:
            singleResult = True
            times = [times]
        
        # 确保时间列已排序
        # df_sorted = df.sort_values('TimeInUnitySecFromTrialStart')
        if refTimes is not None:
            sorted_times = refTimes
            needInd = True
        else:
            sorted_times = df['TimeInUnitySecFromTrialStart'].values
        sorted_indices = df.index.values
        
        times_query = np.asarray(times)
        results = []
        
        for t in times_query:
            # 计算允许的时间范围
            low = t - tolerance
            high = t + tolerance
            
            # 找到候选范围的左右边界
            left_idx = np.searchsorted(sorted_times, low, side='left')
            right_idx = np.searchsorted(sorted_times, high, side='right')
            
            if left_idx < right_idx:
                # 在候选范围内找最接近的时间
                candidates = sorted_times[left_idx:right_idx]
                diffs = np.abs(candidates - t)
                min_rel_idx = np.argmin(diffs)
                min_idx = left_idx + min_rel_idx
            else:
                # 如果没有候选，全局找最近
                diffs = np.abs(sorted_times - t)
                min_idx = np.argmin(diffs)
            
            # 获取原始数据框中的索引
            if refTimes is not None:
                original_index = np.searchsorted(refTimes, sorted_times[min_idx], side='left')
            else:
                original_index = sorted_indices[min_idx]
            results.append(original_index)
        
        if needInd:
            return results[0] if singleResult else results
        else:
            return df.loc[results].loc[:, 'x':'y'].values[0] if singleResult else df.loc[results]['x':'y'].values
    
    def SpeedSegment(self, speedArray, posDf, stability_checks = 3, sensitivity = 0.5, min_run_duration = 20):

        baseline = np.median(speedArray)
        mad = median_abs_deviation(speedArray, scale='normal')
        threshold = 0.8 * baseline + sensitivity * mad

        # 状态机参数
        segments = []
        current_start = None
        confirm_counter = 0
        in_motion = False
        
        # 主状态检测循环
        for idx, speed in enumerate(speedArray):
            if speed > threshold:
                confirm_counter = min(confirm_counter + 1, stability_checks)
                
                if not in_motion and confirm_counter >= stability_checks:
                    # 确认运动开始
                    current_start = max(0, idx - stability_checks + 1)
                    in_motion = True
            else:
                if in_motion:
                    confirm_counter = max(confirm_counter - 1, -stability_checks)
                    
                    if confirm_counter <= -stability_checks:
                        # 确认运动结束
                        current_end = idx - 1
                        if current_end - current_start + 1 >= min_run_duration:
                            segments.append( (current_start, current_end) )
                        in_motion = False
                        current_start = None
        
        # 处理最后未闭合段
        if in_motion:
            current_end = len(speedArray) - 1
            if current_end - current_start + 1 >= min_run_duration:
                segments.append( (current_start, current_end) )
        
        # 智能合并相邻段
        merged_segments = []
        prev_start, prev_end = None, None
        
        for seg in segments:
            if not merged_segments:
                merged_segments.append(seg)
                prev_start, prev_end = seg
                continue
                
            current_start, current_end = seg
            
            # 合并条件：间隔时间小于最小静止时长
            if (current_start - prev_end) <= 5:
                new_seg = (prev_start, current_end)
                merged_segments[-1] = new_seg
                prev_start, prev_end = new_seg
            else:
                merged_segments.append(seg)
                prev_start, prev_end = seg
        
        # 生成最终连续分段
        final_segments = []
        last_end = -1
        
        for seg_start, seg_end in merged_segments:
            # 添加前导静止段
            if seg_start > last_end + 1:
                final_segments.append( (last_end + 1, seg_start - 1) )
            
            # 添加运动段
            final_segments.append( (seg_start, seg_end) )
            last_end = seg_end
        
        # 添加最后静止段
        if last_end < len(speedArray) - 1:
            final_segments.append( (last_end + 1, len(speedArray)-1) )
        segDfs = []
        for seg in final_segments:
            segDfs.append(posDf.iloc[seg[0]:seg[1]])
        # DrawTrack(posDf['smooth_x'].values, posDf['smooth_y'].values, speedArray, [i[0] for i in final_segments], addInfo=f"test{np.random.randint(10000)}", show=True, save=False, pointSize = 2)
        return segDfs
        
    
    def SmoothTrack(self):
        self.PosMainDf['smooth_x'] = np.array(self.PosMainDf['x'].values.astype(int))
        self.PosMainDf['smooth_y'] = 1080 - np.array(self.PosMainDf['y'].values.astype(int))
        # x = self.PosMainDf['x'].values.astype(int)
        # y = self.PosMainDf['y'].values.astype(int)

        # distancesToCenter = np.sqrt((x - sceneinfo[0])**2 + (y  - sceneinfo[1])**2)
        # # 创建掩码，排除首行
        # mask = distancesToCenter < sceneinfo[2]
        # mask[0] = True
        # self.PosMainDf['smooth_x'] = self.PosMainDf['x'].where(mask).ffill()
        # self.PosMainDf['smooth_y'] = self.PosMainDf['y'].where(mask).ffill()

        # window_size = 7  # 根据实际情况调整
        # tempx = self.PosMainDf['smooth_x'].rolling(window_size, center=True, min_periods=1).median()
        # tempy = self.PosMainDf['smooth_y'].rolling(window_size, center=True, min_periods=1).median()

        # # 2. 轨迹平滑（Savitzky-Golay滤波器）
        # window_length = 15  # 必须为奇数，根据实际情况调整
        # polyorder = 3       # 多项式阶数
        # self.PosMainDf['smooth_x'] = savgol_filter(tempx, window_length, polyorder)
        # self.PosMainDf['smooth_y'] = savgol_filter(tempy, window_length, polyorder)
        # self.PosMainDf['smooth_y'] = 1080 - np.array(self.PosMainDf['smooth_y'].values) #仅为展示，实际计算应用x, y或者1080 - smoothed_y
        # print("Track smoothed.")

    def QueryTiemProjectToTrialStage(self, query_ts):
    # 将时间序列分割为trials
        weights = [0.1, 0.6, 0.3]
        
        singleResult = False
        if type(query_ts) is not list and type(query_ts) is not np.ndarray:
            singleResult = True
            query_ts = [query_ts]
        
        # 提取阶段开始时间列表用于二分查找
        end_times = self.trialTimeMarkCollected[:, -1]
        results = []
        for query_t in query_ts:
        
            # 使用bisect找到对应的阶段
            idx = bisect.bisect_right(end_times, query_t)
            if idx <= 0 or idx >= len(self.trialTimeMarkCollected):
                results.append(-1)
                continue
            elif idx == 1:
                previous_trial_end = self.trialTimeMarkCollected[0, 0] - 2
            else:
                previous_trial_end = end_times[idx - 1]
            trialTimes = np.insert(self.trialTimeMarkCollected[idx], 0, previous_trial_end)

            phase_type = bisect.bisect_right(trialTimes, query_t) - 1
            # 计算完成比例
            duration = max(0.01, trialTimes[phase_type + 1] - trialTimes[phase_type])
            progress = (query_t - trialTimes[phase_type]) / duration

            results.append(idx + sum(weights[:phase_type]) + progress * weights[phase_type])
        
        if singleResult:
            return results[0]
        else:
            return results
    # def TrackNormalization(posArray:np.ndarray):
    def TrackNormalization(self, trajectory, start_dist_threshold, turn_angle_threshold, max_length=None, initial_window=5):
        """
        处理轨迹数据，计算初始方向并将轨迹转换为极坐标形式。
        
        参数：
        - trajectory: 列表，包含一系列二维坐标点，格式为[[x1, y1], [x2, y2], ...]
        - start_dist_threshold: 浮点数，起始点忽略的距离阈值
        - turn_angle_threshold: 浮点数，转向角度阈值（度）
        - max_length: 整数或None，保留的最大轨迹长度
        - initial_window: 整数，计算初始方向的窗口大小
        
        返回：
        - theta_base: 基准方向（弧度）
        - polar_points: 极坐标轨迹点列表，格式为[[phi1, d1], [phi2, d2], ...]
        """
        if len(trajectory) < 2:
            return None, []
        x0, y0 = trajectory[0]
        i_start = None
        
        # 寻找有效起始点i_start
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - x0
            dy = trajectory[i][1] - y0
            dist = math.hypot(dx, dy)
            if dist >= start_dist_threshold:
                i_start = i
                break
        
        if i_start is None:
            return None, []
        
        # 计算初始方向（基于initial_window个点的平均方向）
        end_initial = min(i_start + initial_window, len(trajectory)-1)
        dx_sum = 0.0
        dy_sum = 0.0
        count = end_initial - i_start + 1
        for i in range(i_start, end_initial + 1):
            dx_sum += trajectory[i][0] - x0
            dy_sum += trajectory[i][1] - y0
        dx_avg = dx_sum / count
        dy_avg = dy_sum / count
        theta_base_initial = math.atan2(dy_avg, dx_avg)
        
        # 寻找第一次转向点
        i_turn = None
        for i in range(end_initial + 1, len(trajectory)):
            dx = trajectory[i][0] - x0
            dy = trajectory[i][1] - y0
            theta_i = math.atan2(dy, dx)
            delta_theta = (theta_i - theta_base_initial + math.pi) % (2 * math.pi) - math.pi
            if abs(delta_theta) >= math.radians(turn_angle_threshold):
                i_turn = i
                break
        
        # 确定基准方向
        if i_turn is not None:
            end_idx = max(i_turn - 1, i_start)
            dx_total = trajectory[end_idx][0] - x0
            dy_total = trajectory[end_idx][1] - y0
        else:
            dx_total = trajectory[-1][0] - x0
            dy_total = trajectory[-1][1] - y0
        theta_base = math.atan2(dy_total, dx_total)
        
        # 转换为极坐标
        polar_points = []
        for point in trajectory[i_start:]:
            dx = point[0] - x0
            dy = point[1] - y0
            r = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            phi = (theta - theta_base + math.pi) % (2 * math.pi) - math.pi
            polar_points.append([phi, r])
        
        # 截断超长轨迹
        if max_length is not None and len(polar_points) > max_length:
            polar_points = polar_points[-max_length:]
        
        return theta_base, polar_points
        

    def DrawTrackPublic(self, _ax = None):
        DrawTrack(self.PosMainDf['smooth_x'].values, self.PosMainDf['smooth_y'].values, self.PosMainDf['speed'], title = "smoothed track", _ax = _ax)

    def ReutrnLickPos(self):
        return self.licksPos.copy()
    def ReturnTotalAccuracy(self):
        return self.trialResults.count(1)/len(self.trialResults)
    def ReturnResults(self):
        return self.trialResults.copy()
    def ReturnElaspedTime(self):
        return self.trialElapsedTime.copy()
    def ReturnElaspedTimeInGroup(self):
        return self.trialElapsedTimeInGroup.copy()
    def ReturnfhElaspedTime(self):
        return self.trialfhElaspedTime.copy()
    def ReturnfhElaspedTimeInGroup(self):
        return self.trialfhElaspedTimeInGroup.copy()
    def ReturnAccuracyInGroup(self):
        return self.trialAccuracyInGroup.copy()
    def ReturnFristHalfDistanceInGroup(self):
        return self.trialfhDistanceInGroup.copy()
    def ReturnDistanceInGroup(self):
        return self.trialfuDistanceInGroup.copy()
    def ReturnSpout(self):
        return self.licksPos.copy()
    def ReturnPosPerTrial(self):
        return self.posPerTrial.copy()
    def returnPosSegPerTrial(self):
        return self.posSegPerTrial.copy()
    def ReturnSpeedPerTrial(self):
        return self.speedPerTrial.copy()
    def ReturnFullUnitDistancePerTrial(self):
        return self.fuDistancePerTrial.copy()
    def ReturnFirstHalfDistancePerTrial(self):
        return self.fhDistancePerTrial.copy()
    def ReturnUnitDistancePerTrial(self):
        return self.uDistancePerTrial.copy()
    def ReturnDistancePerTrial(self):
        return self.DistancePerTrial.copy()
    def ReturntrialkMark(self):
        return self.trialPosMark.copy()
    def ReturnTrialTimeMark(self):
        return self.trialTimeMark.copy()
    # def ReturnTrialDirection(self):
    #     return self.trackDirection.copy()
    # def ReturnTrackPolar(self):
    #     return self.posPolarPerTrial.copy()
    def ReturnRightSpout(self):
        tempRes = self.trialResults * -1 + 1
        return np.abs(self.licksPos - tempRes)
    def ReturnDeviceEvents(self):
        return self.deviceEventRecord.copy()
    def ReturnMean(self):
        return {"trialElapsedTimeMean": self.trialElapsedTimeMean , "trialIntervalTimeMean": self.trialIntervalTimeMean}
        

class MouseDailyRecord:

    class DayProcess(threading.Thread):
        def __init__(self, dayInstance:Day):
            super().__init__()
            self.dayInstance:Day = dayInstance
    
        def run(self):
            if not self.dayInstance.Load():
                self.dayInstance.Process()
    
    def __init__(self, _mouseInd, _RecMainDfs, _PosMainDfs, _config, _trialPerGroup, _eventFiles = [], multiThread = True):
        self.Days:list[Day] = []

        self.mouseInd = _mouseInd
        self.RecMainDfs = _RecMainDfs
        self.PosMainDfs = _PosMainDfs
        self.configs = _config
        self.trialPerGroup = _trialPerGroup
        self.multiThread = multiThread
        
        threads = []
        for day in range(0, len(self.RecMainDfs)):
            tempeventFile = _eventFiles[day] if day < len(_eventFiles) else ''
            self.Days.append(Day(self.RecMainDfs[day], self.PosMainDfs[day], self.mouseInd, day, tempeventFile))
        for day in self.Days:
            if self.multiThread:
                tempThread = self.DayProcess(day)
                tempThread.start()
                threads.append(tempThread)
            else:
                day.Process()  
            # tempThread.join()
        for thread in threads:
            thread.join()

    def ReturnLickPos(self):
        _tempLickPosEveryDay = []
        for day in self.Days:
            _tempLickPosEveryDay.append(day.ReutrnLickPos())
        return _tempLickPosEveryDay

    def ReturnResults(self):
        _tempResultsEveryDay = []
        for day in self.Days:
            _tempResultsEveryDay.append(day.ReturnResults())
        return _tempResultsEveryDay
    
    def ReturnAccuracyInGroup(self):
        _tempResultsInGroupEveryDay = []
        for day in self.Days:
            _tempResultsInGroupEveryDay.append(day.ReturnAccuracyInGroup())
        return _tempResultsInGroupEveryDay
    
    def ReturnElaspedTime(self):
        _tempElapsedTimeEveryDay = []
        for day in self.Days:
            _tempElapsedTimeEveryDay.append(day.ReturnElaspedTime())
        return _tempElapsedTimeEveryDay
        
    def ReturnElaspedTimeInGroup(self):
        _tempElapsedTimeInGroupEveryDay = []
        for day in self.Days:
            _tempElapsedTimeInGroupEveryDay.append(day.ReturnElaspedTimeInGroup())
        return _tempElapsedTimeInGroupEveryDay
    
    def ReturnfhElaspedTime(self):
        _tempfhElapsedTimeEveryDay = []
        for day in self.Days:
            _tempfhElapsedTimeEveryDay.append(day.ReturnfhElaspedTime())
        return _tempfhElapsedTimeEveryDay
    
    def ReturnfhElaspedTimeInGroup(self):
        _tempfhElapsedTimeInGroupEveryDay = []
        for day in self.Days:
            _tempfhElapsedTimeInGroupEveryDay.append(day.ReturnfhElaspedTimeInGroup())
        return _tempfhElapsedTimeInGroupEveryDay
    
    def ReturnFristHalfDistanceInGroup(self):
        _tempFristHalfDistanceInGroupEveryDay = []
        for day in self.Days:
            _tempFristHalfDistanceInGroupEveryDay.append(day.ReturnFristHalfDistanceInGroup())
        return _tempFristHalfDistanceInGroupEveryDay
    
    def ReturnDistanceInGroup(self):
        _tempDistanceInGroupEveryDay = []
        for day in self.Days:
            _tempDistanceInGroupEveryDay.append(day.ReturnDistanceInGroup())
        return _tempDistanceInGroupEveryDay

    def ReturnSpout(self):
        _tempLickPosEveryDay = []
        for day in self.Days:
            _tempLickPosEveryDay.append(day.ReturnSpout())
        return _tempLickPosEveryDay
    
    def ReturnRightSpout(self):
        _tempRightLickPosEveryDay = []
        for day in self.Days:
            _tempRightLickPosEveryDay.append(day.ReturnRightSpout())
        return _tempRightLickPosEveryDay
    
    def ReturnMean(self):
        _tempMean = []
        for day in self.Days:
            _tempMean.append(day.ReturnMean())
        return _tempMean
    
    def ReturnPos(self):
        _tempPosPerTrialEveryDay = []
        for day in self.Days:
            _tempPosPerTrialEveryDay.append(day.ReturnPosPerTrial())
        return _tempPosPerTrialEveryDay

    def returnPosSeg(self):
        _tempPosSegPerTrialEveryDay = []
        for day in self.Days:
            _tempPosSegPerTrialEveryDay.append(day.returnPosSegPerTrial())
        return _tempPosSegPerTrialEveryDay
  
    def ReturnSpeed(self):
        _tempSpeedPerTrialEveryDay = []
        for day in self.Days:
            _tempSpeedPerTrialEveryDay.append(day.ReturnSpeedPerTrial())
        return _tempSpeedPerTrialEveryDay

    def ReturnFullUnitDistance(self):
        _tempFullUnitDistancePerTrialEveryDay = []
        for day in self.Days:
            _tempFullUnitDistancePerTrialEveryDay.append(day.ReturnFullUnitDistancePerTrial())
        return _tempFullUnitDistancePerTrialEveryDay
    
    def ReturnFirstHalfDistance(self):
        _tempFirstHalfUnitDistancePerTrialEveryDay = []
        for day in self.Days:
            _tempFirstHalfUnitDistancePerTrialEveryDay.append(day.ReturnFirstHalfDistancePerTrial())
        return _tempFirstHalfUnitDistancePerTrialEveryDay

    def ReturnUnitDistance(self):
        _tempUnitDistancePerTrialEveryDay = []
        for day in self.Days:
            _tempUnitDistancePerTrialEveryDay.append(day.ReturnUnitDistancePerTrial())
        return _tempUnitDistancePerTrialEveryDay

    def ReturnDistance(self):
        _tempDistancePerTrialEveryDay = []
        for day in self.Days:
            _tempDistancePerTrialEveryDay.append(day.ReturnDistancePerTrial())
        return _tempDistancePerTrialEveryDay

    def ReturnTrialMark(self):
        _tempLickMarkEveryDay = []
        for day in self.Days:
            _tempLickMarkEveryDay.append(day.ReturntrialkMark())
        return _tempLickMarkEveryDay
    
    def ReturnTrialTimeMark(self):
        _tempTimeMarkEveryDay = []
        for day in self.Days:
            _tempTimeMarkEveryDay.append(day.ReturnTrialTimeMark())
        return _tempTimeMarkEveryDay
    
    # def ReturnTrialDirection(self):
    #     _tempDirectionPerTrialEveryDay = []
    #     for day in self.Days:
    #         _tempDirectionPerTrialEveryDay.append(day.ReturnTrialDirection())
    #     return _tempDirectionPerTrialEveryDay
    
    # def ReturnTrackPolar(self):
    #     _tempPolarPerTrialEveryDay = []
    #     for day in self.Days:
    #         _tempPolarPerTrialEveryDay.append(day.ReturnTrackPolar())
    #     return _tempPolarPerTrialEveryDay
    
    def ReturnEvents(self):
        _tempEventsEveryDay = []
        for day in self.Days:
            _tempEventsEveryDay.append(day.ReturnDeviceEvents())
        return _tempEventsEveryDay

    def ReturnRightSpout(self):
        tempRes = self.trialResults * -1 + 1
        return np.abs(self.licksPos - tempRes)
    
def DrawTrack(x, y, speed = None, segment:list = [], title ="", _ax:plt.Axes = None, addInfo = "", show = True, save = False, fig = None, lw = 0.2, pointSize = 0):
    if _ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        outaxes = False
    else:
        ax = _ax
        title = _ax.title
        outaxes = True

    if speed is not None:
        if type(speed) == list or type(speed) == np.ndarray:
            _speed = copy.deepcopy(speed)
            median = np.median(_speed)
            # 计算绝对偏差的中位数（MAD）
            mad = np.median(np.abs(_speed - median))
            # 定义离群值阈值
            cutoff = median + 3 * mad
            # 过滤离群值
            filtered_max = np.max(_speed[_speed <= cutoff])
        elif type(speed) in [int, float]:
            filtered_max = speed
            speed = np.array([speed]*len(x))
            _speed = speed
        else:
            speed = []
            filtered_max = len(x)
            _speed = np.array(range(0, len(x)))

    else:
        speed = []
        _speed = np.array(range(0, len(x)))
        filtered_max = len(x)
    

    segment:np.array = np.array(segment)
# 创建线段集合, 
    if len(x):
        if len(segment):
            if segment[0] > 0:
                segment = np.insert(segment, 0, 0)
            for i_seg in range(len(segment) - 1):
                start = segment[i_seg]
                end = segment[i_seg+1]
                points = np.array([x[start:end], y[start:end]]).T.reshape(-1, 1, 2)
                segments_vis = np.concatenate([points[:-1], points[1:]], axis=1)

                # 创建颜色映射
                norm = plt.Normalize(_speed.min(), filtered_max, clip=True)
                lc = LineCollection(segments_vis, cmap="rainbow", norm=norm, linewidth=lw)
                if len(speed):
                    lc.set_array(np.sqrt(_speed[start:end]))
                else:
                    lc.set_array(np.array(range(0, end - start)))

                ax.add_collection(lc)       
                if pointSize > 0: 
                    plt.scatter(x[start:end], y[start:end], s=pointSize, c='black', alpha=0.5)
            if pointSize > 0: 
                plt.scatter(x[segment[segment > 0]], y[segment[segment > 0]], s=pointSize * 2, c='red', alpha=0.5)
                for i_seg in range(len(segment) - 1):
                    plt.annotate(f"{i_seg}_{segment[i_seg + 1]}", (x[segment[i_seg]], y[segment[i_seg]]), size=10, color='black')
                plt.scatter(x[0], y[0], s=pointSize * 10, c='green', marker='o', alpha=0.5)

        else:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments_vis = np.concatenate([points[:-1], points[1:]], axis=1)

            # 创建颜色映射
            norm = plt.Normalize(_speed.min(), filtered_max, clip=True)
            lc = LineCollection(segments_vis, cmap="rainbow", norm=norm, linewidth=lw)
            lc.set_array(np.sqrt(_speed) if len(speed) else _speed)
            if pointSize > 0: 
                plt.scatter(x, y, s=pointSize, c='black', alpha=0.5)
                plt.scatter(x[0], y[0], s=pointSize * 10, c='green', marker='o', alpha=0.5)

            ax.add_collection(lc)

    if _ax is None:
        plt.rcParams['axes.facecolor'] = 'white'
        ax.set_xlim(0, 1440)
        ax.set_ylim(0, 1080)
        ax.set_aspect(1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        plt.tight_layout()
        if show:
            if len(speed):
                fig.colorbar(lc, ax=ax, label='sqrt Speed', cmap="rainbow")
            plt.show()
            # fig.show()
        if save:
            fig.savefig(outputPath+f'{addInfo}.png', dpi = 600)
    elif fig is not None:
        if len(speed):
            fig.colorbar(lc, ax=ax, label='sqrt Speed', cmap="rainbow")
            # fig.show()
        if save:
            fig.savefig(outputPath+f'{addInfo}.png', dpi = 600)
        # else:
        #     fig.show()
    if not outaxes:
        fig.clear()
        plt.close(fig)
        
outputPath = 'trackPerTrial/'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

import warnings
warnings.filterwarnings("ignore")
multiThread = True
MouseRecords:list[MouseDailyRecord] = []
for mouseIndex in range(0, len(mouseNames)):
    MouseRecords.append(
        # MouseDailyRecord(mouseNames[mouseIndex], AllMouseRecMainDfs[mouseIndex], AllMousePosMainDfs[mouseIndex], configs[mouseIndex], trialPerGroup, AllMouseEventLs[mouseIndex], multiThread=multiThread)
    )