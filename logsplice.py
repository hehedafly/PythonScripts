import pandas as pd
import numpy as np
from tkinter import Tk ,filedialog
from ParseEvent import*

root=Tk()
root.withdraw()

limitedColumns = ['type', 'delta time', 'mode', 'trial', 'lickPos', 'result']

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
                _config = line + "\n"

            elif _addtion is not None:
                _addtion.append(line)

    return _config


cur=list(filedialog.askopenfilenames(filetypes=[('logs', ('.txt', '.evt'))]))
if not cur:
    exit()

cur.sort()
eventfile = [f for f in cur if f.endswith('.evt')]
eventfile = eventfile[0] if len(eventfile) > 0 else None
recfiles = [f for f in cur if f.endswith('rec.txt')]
posfiles = [f for f in cur if f.endswith('pos.txt')]

logevent = ParseEvent(eventfile) if eventfile is not None else None

mainRecDf = None
subRecDf = []
mainPosDf = None
subPosDf = []
for i, recfile in enumerate(recfiles):
    _datarows = []
    _config = readToDataRows(recfile, _datarows)

    recDf = pd.DataFrame(_datarows, columns=limitedColumns)
    recDf = recDf.drop(recDf.index[recDf['type'] == "type"])
    recDf['delta time'] = np.float64(recDf['delta time'])
    if mainRecDf is None:
        mainRecDf = recDf
    else:
        subRecDf.append(recDf)

    _datarows = []
    posDf = pd.read_csv(posfiles[i],sep='\t',names=['x','y', 'syncInd', "100*pythonTime", "frameInd", "TimeInUnitySecFromTrialStart"], dtype=str)
    posDf = posDf.iloc[1:]
    posDf.loc[:,'x':'frameInd'] = posDf.loc[:,'x':'frameInd'].astype(int, errors='ignore')
    posDf.loc['TimeInUnitySecFromTrialStart'] = posDf['TimeInUnitySecFromTrialStart'].astype(float)
    if mainPosDf is None:
        mainPosDf = posDf
    else:
        subPosDf.append(posDf)

trialEventTimeNow = {}
trialStartTimes = []
trialEndTimes = []
trialFinishTimes = []

trialStartTimes = mainRecDf.loc[mainRecDf['type'] == 'start']['delta time'].values
trialEventTimeNow['start'] = np.array(trialStartTimes)
trialEventTimeNow['MSStart'] = mainRecDf.loc[((mainRecDf['type'] == 'miniscopeRecord') & (mainRecDf['lickPos'] != '0'))]['delta time'].values

trialStartAlign = auto_align_with_ratio(
    trialEventTimeNow['start'],
    logevent['VisualStimulate']['Time_E'],
    ratio=50000,
    tolerance=25000
)
trialStartTimeInLogevent = np.asarray(logevent['VisualStimulate']['Time_E'])[np.array(trialStartAlign)]
rescale = (trialStartTimeInLogevent[-1] - trialStartTimeInLogevent[0]) / (trialEventTimeNow['start'][-1] - trialEventTimeNow['start'][0])
offset = trialStartTimeInLogevent[0] - trialEventTimeNow['start'][0]*rescale

miniscopeRecordTimeInUnity = AlignProjectBackToUnityTime(logevent['MsEnable'], offset, rescale)
subminiscopeRecordTimeInUnity = miniscopeRecordTimeInUnity[np.setdiff1d(range(len(miniscopeRecordTimeInUnity)), np.array(auto_align_with_ratio(trialEventTimeNow['MSStart'], miniscopeRecordTimeInUnity, 1, 10)))]

lastFrameInd = mainPosDf['frameInd'].max()
lastSyncFrameInd = mainPosDf['syncInd'].max()
for i, subRec in enumerate(subRecDf):
    startInd = subRec.loc[subRec['type'] == 'miniscopeRecord'].loc[subRec['lickPos'] != '0'].index[0]
    endInd = subRec.loc[subRec['type'] == 'miniscopeRecord'].loc[subRec['lickPos'] == '0'].index[-1]
    startTime = subRec.loc[startInd]['delta time']
    endTime = subRec.loc[endInd]['delta time']

    recordRecBlocks = subRec.loc[startInd:endInd]
    timeOffset = subminiscopeRecordTimeInUnity[0] - startTime
    recordRecBlocks['delta time'] = recordRecBlocks['delta time'] + timeOffset
    mainRecDf = pd.concat([mainRecDf, recordRecBlocks])
    subminiscopeRecordTimeInUnity = subminiscopeRecordTimeInUnity[len(subRec.loc[subRec['type'] == 'miniscopeRecord'].loc[subRec['lickPos'] != '0']):]
    recordPosBlocks = subPosDf[i]

    recordPosBlocks = recordPosBlocks.loc[recordPosBlocks['delta time'] >= startTime]
    recordPosBlocks = recordPosBlocks.loc[recordPosBlocks['delta time'] <= endTime + 1]
    recordPosBlocks['frameInd'] += lastFrameInd
    recordPosBlocks['syncInd'] += lastSyncFrameInd
    lastFrameInd = recordPosBlocks['frameInd'].max()
    lastSyncFrameInd = recordPosBlocks['syncInd'].max()

    mainPosDf = pd.concat([mainPosDf, recordPosBlocks])
    # MSFrameTime = AlignProjectBackToUnityTime(eventDict['MsSync']['Time_E'], offset, rescale)
    # correspondPosDfIndex = DayClassInstance.GetPosOrIndAtTime(MSFrameTime, needInd=True)

    # peakIndexes = AllMouseMSFrameInPosDfIndex[mouseInd][day]
    # trialEventTimeNow = trialEventTime[mouseInd][day]

    # dropcount = len(peakIndexes) - len(peakIndexes[peakIndexes > 0])
    # peakIndexes = peakIndexes[peakIndexes > 0]
    # peakTime = dayInstance.PosMainDf.loc[peakIndexes]['TimeInUnitySecFromTrialStart'].values
    # tempStage = dayInstance.QueryTiemProjectToTrialStage(peakTime)
    # tempFracStage = np.array(tempStage) - np.array(tempStage, dtype= int)
    # stageRelatedTrial = np.array(tempStage, dtype= int)
    # randomSelectedTrials = np.random.choice(np.unique(stageRelatedTrial), int(len(np.unique(stageRelatedTrial)) / 1.5), replace=False)
    # tempSpeed = dayInstance.PosMainDf.loc[peakIndexes]['speed'].values
    # relatedPos = dayInstance.PosMainDf.loc[peakIndexes][['x','y']].values
    passedFrame = startTime + timeOffset

    


