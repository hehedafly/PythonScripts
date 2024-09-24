import os
from math import*
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

ignoreFristTrials = 30
global trialGroupCount, trialPerGroup
trialGroupCount= -1
trialPerGroup = 100
mouseInds = ["A24", "A25", "A30", "A31"]
LogsFolders = ["./A24", "./A25", "./A30", "./A31"]
MouseRecords = []
configs = []

class MouseDailyRecord:
    def __init__(self, _mouseInd, _MainDfs, _config, _trialPerGroup):
        self.Days = []

        self.mouseInd = _mouseInd
        self.MainDfs = _MainDfs
        self.configs = _config
        self.trialPerGroup = _trialPerGroup

        for day in range(0, len(AllMouseMainDfs)):
            self.Days.append(self.Day(self.MainDfs[day], self.mouseInd, day))
        for day in self.Days:
            day.Process()    

    def ReturnResults(self):
        _tempResultsEveryDay = []
        for day in self.Days:
            _tempResultsEveryDay.append(day.ReturnResults())
        return _tempResultsEveryDay


    class Day:
        def __init__(self, _MainDf, _name, _day):
            self.trials                  = []
            self.startIndices            = []
            self.EndIndices              = []
            self.licks                   = []
            self.lickInterval            = []
            self.lickIntervalLog         = []
            self.trialResults            = []
            self.trialElapsedTime        = []
            self.trialIntervalTime       = []
            self.trialResultsInGroup     = []
            self.trialElapsedTimeInGroup = []
            self.trialIntervalInGroup    = []

            self.MainDf = _MainDf
            self.trialPerGroup = -1
            self.day = _day
            self.name = _name

        def Process(self): 
            init = self.MainDf.index[self.MainDf['type'] == "init"]
            if (len(init)):
                init = max(init)
            else:
                init = 0
            self.MainDf = self.MainDf[init:]

            self.startIndices    =          self.MainDf.index[self.MainDf['type'] == "start"]#trial处理均以开始结束Index为标准
            self.EndIndices      =          self.MainDf.index[self.MainDf['type'] == "end"]
            self.licks           = np.array(self.MainDf.loc[self.MainDf["type"] == "lick"]["delta time"])
            self.startIndices.delete(range(len(self.EndIndices), len(self.startIndices)))
            
            for i in range(1, len(self.startIndices)):
                self.trials.append(             self.MainDf.loc[self.startIndices[i-1] : self.startIndices[i], :])
                self.trialResults.append(       self.trials[i - 1].loc[self.EndIndices[i - 1]]["result"])
                self.trialElapsedTime.append(   self.trials[i - 1].loc[self.EndIndices[i-1]]['delta time'] - self.trials[i - 1].loc[self.startIndices[i-1]]['delta time'])
                self.trialIntervalTime.append(  self.trials[i - 1].loc[self.startIndices[i]]['delta time'] - self.trials[i - 1].loc[self.EndIndices[i-1]]['delta time'])
                
            self.trials             = self.trials[              ignoreFristTrials :]
            self.trialResults       = self.trialResults[        ignoreFristTrials :]
            self.trialElapsedTime   = self.trialElapsedTime[    ignoreFristTrials :]
            self.trialIntervalTime  = self.trialIntervalTime[   ignoreFristTrials :]

            self.trialResults = np.int8(self.trialResults)

            self.lickInterval = self.licks[1:] - self.licks[0: -1]
            self.lickInterval = self.lickInterval[self.lickInterval > 0]
            self.lickIntervalLog = np.log(1/self.lickInterval)

            if(trialGroupCount > 0):
                self.trialPerGroup = ceil(len(self.trials)/trialGroupCount)
            else:
                self.trialPerGroup = trialPerGroup

            for i in range(0, ceil(len(self.trialResults) / self.trialPerGroup)):
                self.trialResultsInGroup.append(    np.sum(self.trialResults[       i:  min(len(self.trialResults)      , i + self.trialPerGroup)]) / self.trialPerGroup)
                self.trialElapsedTimeInGroup.append(np.sum(self.trialElapsedTime[   i:  min(len(self.trialElapsedTime)  , i + self.trialPerGroup)]) / self.trialPerGroup)
                self.trialIntervalInGroup.append(   np.sum(self.trialIntervalTime[  i:  min(len(self.trialIntervalTime) , i + self.trialPerGroup)]) / self.trialPerGroup)

            self.trialElapsedTimeMean    = np.mean(self.trialElapsedTime)
            self.trialIntervalTimeMean   = np.mean(self.trialIntervalTime)

            interferentialTrials = self.trialResults[self.trialResults < 0]
            if(len(interferentialTrials) > 0):
                self.trials            = np.delete(self.trials           , interferentialTrials)
                self.trialResults      = np.delete(self.trialResults     , interferentialTrials)
                self.trialElapsedTime  = np.delete(self.trialElapsedTime , interferentialTrials)
                self.trialIntervalTime = np.delete(self.trialIntervalTime, interferentialTrials)

        def ReturnTotalAccuracy(self):
            return self.trialResults.count(1)/len(self.trialResults)
        def ReturnResults(self):
            return self.trialResults.copy()



    
fileList = []
for mouseIndex in LogsFolders:
    fileList.append(os.listdir(mouseIndex))
    fileList[-1].sort()

def readToDataRows(_name, _dataRows):
    _config = ""
    with open(_name, 'r') as file:
        for line in file:
            if line.strip() and "\t" in line:
                # 按制表符分割行，并添加到数据行列表中
                _dataRows.append(line.strip().split('\t'))
                #if len(dataRows) > 1:
                    #dataRows[-1][1] = float(dataRows[-1][1])
                #dataRowsTyped.append(dataRows[-1])
            elif line.startswith("{"):
                _config = line
    _dataRows.pop(0)
    return _config
            
AllMouseMainDfs = []
AllDays = []
for mouseIndex in range(0, len(fileList)):#fileList按mouse排序
    AllMouseMainDfs.append([])
    AllDays.append(len(fileList[mouseIndex]))
    configs.append([])
    for DailyRecord in fileList[mouseIndex]:
        _datarows = []
        _config = readToDataRows(LogsFolders[mouseIndex] + "/" + DailyRecord, _datarows)
        configs[-1].append(_config)
        AllMouseMainDfs[-1].append(pd.DataFrame(_datarows, columns=[
            'type', 'delta time', 'mode', 'trial', 'lickPos', 'result'
        ]))
        AllMouseMainDfs[-1][-1]['delta time'] = np.float64(AllMouseMainDfs[-1][-1]['delta time'])


for mouseIndex in range(0, len(mouseInds)):
    MouseRecords.append(
        MouseDailyRecord(mouseInds[mouseIndex], AllMouseMainDfs[mouseIndex], configs[mouseIndex], trialPerGroup)
    )



#print(MainDf.loc[0:2, :])


# fig, axes = plt.subplots(1, 3, figsize=(8,5))
# axelickInterval = axes[0]
# axeResult = axes[1]
# axeTrialIntervalTime = axes[2]
#fig, axes = plt.subplots(max(AllDays), len(mouseInds))
ResultsEverydayEveryMouse = []
for mouse in MouseRecords:
    ResultsEverydayEveryMouse.append(mouse.ReturnResults())

fig, axes = plt.subplots(2, len(mouseInds))

# for i in range (len(mouseInds)):

# axeResult = axes[0]
# axeResultPerLickPot = axes[1]
# n, logInterval, _ = axelickInterval.hist(lickInterval, bins= 20, rwidth= 0.9)
# logIntervalMainValue = logInterval[1:][n == max(n)][0]
# CommonLickInterval = pow(10, logIntervalMainValue)


# axeResult.plot(range(0, len(trialResultsInGroup)), trialResultsInGroup)
# axisInterval = axeResult.twinx()
# axisInterval.plot(range(0, len(trialElapsedTimeInGroup)), trialElapsedTimeInGroup, color = 'g')
# #fig.subplots(121)
# fig.legend(["Accuracy", "trial speed"])


# plt.show()
# 显示DataFrame的前几行以检查数据
# print(MainDf.head()) 