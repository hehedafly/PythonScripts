from tkinter import Tk ,filedialog
from MouseRecord.mouseTrackPlotFunc import*
# from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
# from scipy.spatial import ConvexHull
# from scipy.interpolate import interp1d
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import random
from matplotlib import pyplot as plt
import json
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from sklearn.utils import resample
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

def CreateDayInstance(summaryLogPath:str = "", neuronDataPath:str = ""):
    cur = filedialog.askopenfilename(filetypes=[('summary_logs', ('.pkl'))]) if (not summaryLogPath or not os.path.exists(summaryLogPath)) else summaryLogPath
    if not cur:
        print("invalid file path")
        return None, None
    savePath = os.path.dirname(cur)
    fileName = os.path.basename(cur)
    #neuronData放在summary同级的对应名称天数文件夹内，文件夹命名格式f"{name}_day{day}"
    _, _, name, day = GetFormatedFileInfo(fileName, extraSuffix="_summary")
    dayInstance = Day('', '', name, day, _savePath = savePath)
    dayInstance.Load()
    eventDict = dayInstance.fullEventRecord
    tempPkl = pd.read_pickle(cur)
    loadedRescale = tempPkl['addons'].values[8]
    if type(loadedRescale) == str and len(loadedRescale):
        loadedRescale = float(loadedRescale)
        loadedOffset = float(tempPkl['addons'].values[9])
    else:
        loadedOffset = tempPkl['addons'].values[9]

    rescale = loadedRescale
    offset = loadedOffset

    trialEventTimeNow = {}
    trialStartTimes = []
    trialEndTimes = []
    trialFinishTimes = []
    for i, tempDf in enumerate(dayInstance.trials):
        # tempDf:pd.DataFrame = MouseRecords[mouseInd].Days[day].trials[trialInd]
        _startTime = tempDf.loc[tempDf['type'] == 'start']['delta time'].values[0]
        _endTime = tempDf.loc[tempDf['type'] == 'end']['delta time'].values[0]
        _finishTime = tempDf.loc[(tempDf['type'] == 'stay')]['delta time'].values[0]
        trialStartTimes.append(_startTime)
        trialEndTimes.append(_endTime)
        trialFinishTimes.append(_finishTime)
    tempDeviceEvent = dayInstance.ReturnDeviceEvents()
    trialEventTimeNow['start'] = np.array(trialStartTimes)
    trialEventTimeNow['startInd'] = np.array(dayInstance.trialIndexes)
    trialEventTimeNow['end'] = np.array(trialEndTimes)
    trialEventTimeNow['finish'] = np.array(trialFinishTimes)
    OGDf = tempDeviceEvent['OGManipulate']
    trialEventTimeNow['OGStart'] = OGDf.loc[OGDf['lickPos'] != '0']['delta time'].values if len(OGDf) > 0 else []
    MSDf = tempDeviceEvent['miniscopeRecord']
    trialEventTimeNow['MSStart'] = MSDf.loc[MSDf['lickPos'] != '0']['delta time'].values if len(MSDf) > 0 else []

    neuronDataPath = filedialog.askopenfilename(filetypes=[('neuronData', ('.pkl'))]) if (not summaryLogPath or not neuronDataPath or not os.path.exists(neuronDataPath)) else neuronDataPath
    return cur, DataPerDay(dayInstance, eventDict, rescale, offset, trialEventTimeNow, neuronDataPath)


class DataPerDay:
    def __init__(self, dayInstance:Day, eventDict:dict, rescale:float, offset:float, trialEvent:dict, neuronDataPath: str, extracted:bool = True, deconvolvedNeuronDataPath:str = None):
        self.name = dayInstance.name
        self.day = dayInstance.day
        self.dayInstance = dayInstance
        self.eventDict = eventDict
        self.rescale = rescale
        self.offset = offset
        self.trialEvent = trialEvent
        self.extracted = extracted
        self.MSFrameInPosDfIndexOringinal = None
        self.MSFrameInPosDfIndexExtracted = None
        self.RelatedBehaviorData:dict = {}
        self.ResampleClusteredData:dict = {}
        self.MSFrameTimeSegmented:np.ndarray = None#以unityTime为基础的Miniscope帧时间片段
        self.MSFrameTimeSegmentedExtracted:np.ndarray = None
        self.sceneInfo = json.loads(self.dayInstance.PosMainDf['sceneInfo'].values[0])
        self.selectedAreas = json.loads(self.dayInstance.PosMainDf['sceneInfo'].values[1])
        self.neuronData:np.ndarray = None
        self.neuronIndex:np.ndarray = None
        self.neuronDataDeconvolved:np.ndarray = None
        
        if not neuronDataPath or not os.path.exists(neuronDataPath):
            print("Neuron data not seleted")
            # neuronData = None
        else:
            infile = open(neuronDataPath, 'rb')
            neuronData = pickle.load(infile)
            infile.close()
            self.neuronIndex = np.array([i for i, x in enumerate(neuronData) if isinstance(x, np.ndarray)])
            self.neuronData = np.vstack(neuronData[self.neuronIndex])
            self.Calc()
            self.ProcessNeuronData(os.path.dirname(neuronDataPath))

        if not deconvolvedNeuronDataPath or not os.path.exists(deconvolvedNeuronDataPath):
            deconvolvedNeuronData = None
        else:
            infile = open(deconvolvedNeuronDataPath, 'rb')
            deconvolvedNeuronData = pickle.load(infile)
            infile.close()
            self.neuronDataDeconvolved = np.vstack(deconvolvedNeuronData[self.neuronIndex])
        
        
    def ExtractFullFrameToNeuronDataShape(self, _array, useDeconvolved:bool = False):
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData
        _array = _array[::2]
        if len(_array) < neuronData.shape[1]:
            raise ValueError(f"array length less than neuron data length, extracted data length:{len(_array)}, neuron data length:{neuronData.shape[1]}")
        return _array[:neuronData.shape[1]]
    
    def GetCorrespondNeuronDataIndex(self, _frameTime:np.ndarray):
        '''_frameTime: Unity时间点'''
        oFrame = self.MSFrameTimeSegmentedExtracted if self.extracted else self.MSFrameTimeSegmented

        pos = np.searchsorted(oFrame, _frameTime, side='left')
        candidates = np.clip(np.stack([pos - 1, pos]), 0, len(oFrame) - 1).T
        chosen = candidates[np.arange(len(_frameTime)), 
                            np.abs(_frameTime[:, None] - oFrame[candidates]).argmin(axis=1)]
        return np.unique(chosen)
        
    def GetCorrespondPosDfIndexWithOffset(self, offset:float, _frameTime:np.ndarray = None):
        '''返回时间点在PosDf中对应index'''
        if _frameTime is None:
            _frameTime = self.MSFrameTimeSegmented
        _ind = self.dayInstance.GetPosOrIndAtTime(_frameTime + offset, needInd=True)
        for _icorr in range(1, len(_ind)):
            _ind[_icorr] = max(_ind[_icorr], _ind[_icorr - 1] + 1) if _ind[_icorr] != 0 else 0
        return _ind
    
    def GetBaslerFrameIndCorrespondPosDfIndexWithOffset(self, offset, _frameTime = None):
        correspondPosDfIndex = self.GetCorrespondPosDfIndexWithOffset(offset, _frameTime)
        correspondBaslerFrameIndex = self.dayInstance.PosMainDf.loc[correspondPosDfIndex]['frameInd'].values
        if correspondPosDfIndex[0] == 0:#之前记录时背景神经元部分可能没有记录到pos文件内
            validInd = np.array(correspondPosDfIndex)
            validInd = validInd[validInd > 0]
            dropedFrameCount = len(correspondPosDfIndex) - len(validInd)
            correspondBaslerFrameIndex[:dropedFrameCount] = 0#丢失的帧设为0
        return correspondBaslerFrameIndex, dropedFrameCount
        
    def GetRelatedPos(self, _indexes = None):
        if _indexes is None:
            _indexes = self.MSFrameInPosDfIndexOringinal
        return self.dayInstance.PosMainDf.loc[_indexes][['x','y']].values

    def Calc(self):
        if self.neuronData is None:
            return
            
        msSyncTick = np.array(self.eventDict['MsSync']['Time_S'] + self.eventDict['MsSync']['Time_E'])
        msSyncTick.sort()
        MSFrameTime = AlignProjectBackToUnityTime(msSyncTick, self.offset, self.rescale)
        MSFrameTimeSegmented = auto_segment(MSFrameTime, 30)
        MSFrameTimeSegmented = [MSFrameTimeSegmented[i] for i in range(len(MSFrameTimeSegmented)) if len(MSFrameTimeSegmented[i]) > 10]
        NeuronDataFrameCount = len(self.neuronData[0])
        NeuronDataDuration = round(NeuronDataFrameCount / 600)
        if round(len(MSFrameTime) / 1200) != NeuronDataDuration:
            while True:
                selectedStr = input(f"there are {len(MSFrameTimeSegmented)} video recorded, each one last for {[round(len(MSFrameTimeSegmented[i]) / 1200) for i in range(len(MSFrameTimeSegmented))]} min, {NeuronDataDuration} min processed, select format: 0,2,4,6")
                selected = np.array(selectedStr.split(','), int)
                if (min(selected) > 0 and max(selected < len(MSFrameTimeSegmented))):
                    selectedFrame = np.array([np.array(MSFrameTimeSegmented[i]) for i in selected]).flatten()
                    if round(len(selectedFrame) / 1200) == NeuronDataDuration:
                        self.MSFrameTimeSegmented = selectedFrame
                        break
                    else:
                        print("selected duration not match")
                else:
                    print("invalid input")
        else:
            self.MSFrameTimeSegmented = np.concatenate(MSFrameTimeSegmented)
        self.MSFrameTimeSegmentedExtracted = self.ExtractFullFrameToNeuronDataShape(self.MSFrameTimeSegmented)
        correspondPosDfIndex = self.GetCorrespondPosDfIndexWithOffset(0)
        # correspondBaslerFrameIndex = self.dayInstance.PosMainDf.loc[correspondPosDfIndex]['frameInd'].values
        # if correspondPosDfIndex[0] == 0:#之前记录时背景神经元部分可能没有记录到pos文件内
        #     validInd = np.array(correspondPosDfIndex)
        #     validInd = validInd[validInd > 0]
        #     dropedFrameCount = len(correspondPosDfIndex) - len(validInd)
        #     correspondBaslerFrameIndex[:dropedFrameCount] = 0#丢失的帧设为0

        self.MSFrameInPosDfIndexOringinal = np.array(correspondPosDfIndex)
        self.MSFrameInPosDfIndexExtracted = self.ExtractFullFrameToNeuronDataShape(self.MSFrameInPosDfIndexOringinal)[:self.neuronData.shape[1]]
        # self.neuronDataSplitPoint = self.refine_stitch_point(self.neuronData, np.add.accumulate([round(len(fs) / 2) for fs in MSFrameTimeSegmented[:-1]]))
        # _, axes = plt.subplots(1, ceil(len(self.neuronIndex) / 10), figsize=(180, 40), dpi = 300)
        # for column in range(ceil(len(self.neuronIndex) / 10)):
        #     [axes[column].plot([100 * i + 50]*2, [0, 1500], c = 'red', lw = 1, alpha = 0.5) for i in range(len(MSFrameTimeSegmented) - 1)]
        #     for row in range(10):
        #         if (column * 10 + row) < len(self.neuronIndex):
        #             drawData = np.array([self.neuronData[column * 10 + row , sp - 50: sp + 50] for sp in np.add.accumulate([round(len(fs) / 2) for fs in MSFrameTimeSegmented[:-1]])]).flatten()
        #             drawData = drawData + row * 150
        #             [axes[column].plot([sp*100 + 50 - 5, sp*100 + 50 + 5], [drawData[sp * 100 + 50]] * 2, lw = 1, c= 'r') for sp in range(len(MSFrameTimeSegmented) - 1)]
        #             axes[column].scatter(np.arange(len(drawData)), drawData, s = 1, alpha = 0.5)
        # plt.tight_layout()
        # plt.savefig("test.png")
    
    def ProcessNeuronData(self, headOrientationsFolderPath:str):
        peakIndexes = self.MSFrameInPosDfIndexExtracted
        dropcount = len(peakIndexes) - len(peakIndexes[peakIndexes > 0])
        peakIndexes = peakIndexes[peakIndexes > 0]
        peakTime = self.dayInstance.PosMainDf.loc[peakIndexes]['TimeInUnitySecFromTrialStart'].values
        tempStage = self.dayInstance.QueryTiemProjectToTrialStage(peakTime)
        tempFracStage = np.array(tempStage) - np.array(tempStage, dtype= int)
        stageRelatedTrial = np.array(tempStage, dtype= int)
        # randomSelectedTrials = np.random.choice(np.unique(stageRelatedTrial), int(len(np.unique(stageRelatedTrial)) / 1.5), replace=False)
        tempSpeed = self.dayInstance.PosMainDf.loc[peakIndexes]['speed'].values
        relatedPosOringinal = self.dayInstance.PosMainDf.loc[peakIndexes][['x','y']].values
                
        hofiles = os.listdir(headOrientationsFolderPath)
        hofiles.sort()

        angleData = None
        angleDeltaData = None
        filepath = os.path.join(headOrientationsFolderPath, "oHeadOrientationInScene.txt")
        oHeadOrientationInScene = np.zeros(4)
        if os.path.exists(filepath):
            with open(filepath) as f:
                oHeadOrientationInScene = np.array(f.read().split(','))
        try:
            hofiles = [getFilesInFolder(os.path.join(headOrientationsFolderPath, hofile), ".csv") for hofile in hofiles]
            hofiles = [hofile[0] for hofile in hofiles if len(hofile)]
        except:
            hofiles = []
            
        if len(hofiles) > 1:
            angleDf = pd.read_csv(hofiles[0], dtype=float, header = 0, names = ['t', 'w', 'x', 'y', 'z'])
            for hofile in hofiles[1:]:
                angleDf = pd.concat([angleDf, pd.read_csv(hofile, dtype=float, header = 0, names = ['t', 'w', 'x', 'y', 'z'])])
            if self.extracted:
                angleDf = self.ExtractFullFrameToNeuronDataShape(angleDf)
            qdata = np.array(list(angleDf[['x', 'y', 'z', 'w']].values), dtype= float)
            qdata = qdata - oHeadOrientationInScene
            rotations = R.from_quat(qdata)
            euler_angles = rotations.as_euler('xyz', degrees=False)
            angleData = np.rad2deg(euler_angles)

            # angleData = angleData[[i % 2 == 0 for i in range(len(angleData))]][dropcount:]
            angleDeltaData = np.insert((angleData[1:] + 180 - angleData[:-1]) % 360 - 180, 0, np.zeros_like(angleData[0]), axis = 0)

        self.neuronData = self.neuronData[:, dropcount:]
        self.neuronDataDeconvolved = self.neuronDataDeconvolved[:, dropcount:]

        # for i in range(len(undeconvedneurondata)):
        #     undeconvedneurondata[i] = undeconvedneurondata[i][dropcount:]
        # undeconvedneurondata = np.array(list(undeconvedneurondata))
        # relatedPos = np.array(relatedPos)

        # fullNeuronInfoDf = pd.DataFrame(self.neuronData.T)
        # fullNeuronInfoDf['peakIndex'] = peakIndexes
        # fullNeuronInfoDf['time'] = peakTime
        # fullNeuronInfoDf.index = peakIndexes
        # # fullNeuronInfoDf['tempFracStage'] = tempFracStage
        # fullNeuronInfoDf.to_csv('neuronInfo.csv', index=True, header=True)
        self.RelatedBehaviorData['tempStage'] = tempStage
        self.RelatedBehaviorData['tempFracStage'] = tempFracStage
        self.RelatedBehaviorData['stageRelatedTrial'] = stageRelatedTrial
        # self.RelatedBehaviorData['randomSelectedTrials'] = randomSelectedTrials
        self.RelatedBehaviorData['tempSpeed'] = tempSpeed
        self.RelatedBehaviorData['angleData'] = angleData
        self.RelatedBehaviorData['angleDeltaData'] = angleDeltaData
        self.RelatedBehaviorData['relatedPos'] = relatedPosOringinal
        self.RelatedBehaviorData['peakTime'] = peakTime
        self.RelatedBehaviorData['peakIndexes'] = peakIndexes
        self.RelatedBehaviorData['neuronData'] = self.neuronData
        self.RelatedBehaviorData['neuronDataDeconvolved'] = self.neuronDataDeconvolved

    def GetNeuronDataIndexByNeuronIndex(self, neuronIndex:np.ndarray) -> np.ndarray:
        return np.array([int(np.where(self.neuronIndex == id)[0][0]) for id in neuronIndex if id in self.neuronIndex])

    def FindBestSingleNeuronFiringMap(self, neuronIndex: int, use_parallel: bool = True, useDeconvolved:bool = False):
        assert neuronIndex in self.neuronIndex, "Invalid neuron index"

        sceneInfo = self.sceneInfo
        neuron_pos = int(np.where(self.neuronIndex == neuronIndex)[0][0])
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData
        neuron_activity = neuronData[neuron_pos]

        center = np.array(sceneInfo[0:2])
        radius = sceneInfo[2]
        chunknum = 21
        chunkScale = np.ceil(float(radius * 2) / 10) * 10 / chunknum
        transformed_x = center[0] - chunkScale * chunknum * 0.5
        transformed_y = center[1] - chunkScale * chunknum * 0.5

        all_pos_indexes = self.MSFrameInPosDfIndexOringinal
        all_pos = self.GetRelatedPos(all_pos_indexes)

        chunked_all = np.array([
            (all_pos[:, 0] - transformed_x) / chunkScale,
            (all_pos[:, 1] - transformed_y) / chunkScale
        ], dtype=int).T
        chunked_all = np.clip(chunked_all, 0, chunknum-1)

        position_prior_map = np.zeros(chunknum * chunknum)
        for coord in chunked_all:
            key = (chunknum - 1 - coord[1]) * chunknum + coord[0]
            if key < len(position_prior_map):
                position_prior_map[key] += 1

        position_prior_map = gaussian_filter(position_prior_map.reshape(chunknum, chunknum), sigma=1)
        position_prior_map = position_prior_map / np.sum(position_prior_map)
        # plt.imshow(position_prior_map.reshape(chunknum, chunknum))
        
        offsets = np.arange(-0.8, 0.8, 0.1)

        # ⚡ 并行执行每个 offset
        if use_parallel and len(offsets) > 1:
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(_process_single_offset)(
                    offset,
                    neuron_activity,
                    chunknum,
                    self.GetCorrespondPosDfIndexWithOffset,  # 传入方法
                    self.GetRelatedPos,                     # 传入方法
                    transformed_x,
                    transformed_y,
                    chunkScale, 
                    position_prior_map
                )
                for offset in offsets
            )
        else:
            # 串行执行（用于调试或小数据）
            results = [
                _process_single_offset(
                    offset,
                    neuron_activity,
                    chunknum,
                    self.GetCorrespondPosDfIndexWithOffset,
                    self.GetRelatedPos,
                    transformed_x,
                    transformed_y,
                    chunkScale,
                    position_prior_map
                )
                for offset in offsets
            ]

        # 过滤失败结果
        offset_results = [r for r in results if r is not None]

        if not offset_results:
            raise RuntimeError("所有 offset 计算均失败")

        best_result = max(offset_results, key=lambda x: x['score'])
        offset_results.sort(key=lambda x: x['score'], reverse=True)
        return {
            'best_offset': best_result['offset'],
            # 'best_map': best_result['map'],
            'best_stability': best_result['stability'],
            'best_concentration': best_result['concentration'],
            'best_score': best_result['score'],
            'all_results': offset_results,
        }

    def ShowSingleNeuronFiringMap(self, assignedCellId:list[int] = None, timeOffset:float = 0, fillWithRandomCells:bool = True, cutPeakAtReward:bool =True, useDeconvolved = False):
        sceneInfo = self.sceneInfo
        # selectedAreas = json.loads(self.dayInstance.PosMainDf['sceneInfo'].values[1])
        # relatedPos = self.RelatedBehaviorData['relatedPos']
        # tempStage = self.RelatedBehaviorData['tempFracStage']
        tempFracStage = self.RelatedBehaviorData['tempFracStage']
        # tempSpeed = self.RelatedBehaviorData['tempSpeed']
        angleData = self.RelatedBehaviorData['angleData']
        # angleDeltaData = self.RelatedBehaviorData['angleDeltaData']
        # peakTime = self.RelatedBehaviorData['peakTime']
        peakIndexes = self.RelatedBehaviorData['peakIndexes'] if timeOffset == 0 else self.GetCorrespondPosDfIndexWithOffset(timeOffset)
        relatedPos = self.RelatedBehaviorData['relatedPos'] if timeOffset == 0 else self.GetRelatedPos(peakIndexes)
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData

        # undeconvedneurondata = self.RelatedBehaviorData['undeconvedneurondata']
        center = sceneInfo[0:2]
        radius = sceneInfo[2]
        chunknum = 21

        chunkScale = np.ceil(float(radius * 2) / 10) * 10 / chunknum
        transformed_x = center[0] - chunkScale * chunknum * 0.5
        transformed_y = center[1] - chunkScale * chunknum * 0.5

        chunkedposdata = np.array([(relatedPos[:, 0] - transformed_x) / chunkScale, (relatedPos[:, 1] - transformed_y) / chunkScale], dtype= int).T

        index_map = {}
        for idx, coord in enumerate(chunkedposdata):
            key = (chunknum - 1 - coord[1]) * chunknum + coord[0]
            if key not in index_map:
                index_map[key] = []
            index_map[key].append(idx)
        
        rows = 2
        fig, axes = plt.subplots(rows * 2, 10, figsize=(100, rows * 20), dpi = 200)
        randIndexes = []
        temp_neurondata = neuronData
        if (assignedCellId is not None):
            if len(assignedCellId)== 0:
                print("No cell selected")
                return
            assignIndex =  self.GetNeuronDataIndexByNeuronIndex(assignedCellId)
            assignIndex = assignIndex[0:20]
        else:
            assignIndex = random.sample(range(0, len(temp_neurondata)), 20)

        while fillWithRandomCells and len(assignIndex) < 20:
            temp = random.sample(range(0, len(temp_neurondata)), 20 - len(assignIndex))
            assignIndex += [id for id in temp if id not in assignIndex]

        for i, ind in enumerate(assignIndex):
        
            tempAxeHeat:plt.Axes = axes[int(i / 10) * rows][i % 10]
            tempAxeshist:plt.Axes = axes[int(i / 10) * rows + 1][i % 10]
            
            randIndexes.append(ind)
            peak = np.array(temp_neurondata[ind]) / np.percentile(temp_neurondata[ind], 99)
            if cutPeakAtReward:
                peak[tempFracStage < 0.05] = 0
            # peak = np.array(unconvedneurondata[ind]) / np.percentile(unconvedneurondata[ind], 99)
            peak[peak > 1] = 1
            peak[peak < 0] = 0
            peak = (peak ** 2)
            peak = (peak - min(peak)) / (max(peak) - min(peak))

            clustedActivityData = np.zeros((chunknum ** 2))
            for k in index_map:
                activity = peak[index_map[k]]
                # activity = activity[activity > 0.5]
                clustedActivityData[k] = sum(activity) / sqrt(len(index_map[k]))
            # secondActivity = max(clustedActivityData[clustedActivityData != int((chunknum ** 2 - 1) / 2)])
            # clustedActivityData[int((chunknum ** 2 - 1) / 2)] = min(secondActivity, clustedActivityData[int((chunknum ** 2 - 1) / 2)])

            tempAxeHeat.imshow(clustedActivityData.reshape((chunknum, chunknum)))
            tempAxeHeat.set_title(f'Neuron {self.neuronIndex[ind]}', fontsize = 60)

            normalizedActivity = peak.copy()
            # # normalizedActivity[normalizedActivity > 0.4] = 1
            normalizedActivity[normalizedActivity <= 0.5] = 0
            if angleData is not None:
                tempAxehistTwin = tempAxeshist.twinx()
                tempAxehistTwin.invert_yaxis()
                angleHistCount, _, _ = tempAxehistTwin.hist(angleData[:,2] / 360 + 0.5, bins = 20, rwidth=0.9, weights = normalizedActivity)
                tempAxehistTwin.plot([0.1] * 2, [0, max(angleHistCount) * 2.1], ls = '--', lw= 2, alpha = 1, color = 'r')
                tempAxehistTwin.plot([0.6] * 2, [0, max(angleHistCount) * 2.1], ls = '--', lw= 2, alpha = 1, color = 'r')
                # tempAxescatter.scatter(np.arange(len(normalizedActivity)), normalizedActivity, s = 0.5)
                stageHistCount, _, _ = tempAxeshist.hist(tempFracStage, bins = 10, rwidth = 0.9, weights = normalizedActivity)
                tempAxeshist.set_ylim(0, max(stageHistCount) * 2.1)
                tempAxehistTwin.set_ylim(max(angleHistCount) * 2.1, 0)
            
        plt.tight_layout()
        plt.show()
        return self.neuronIndex[assignIndex]

    def ResampleAndClusterMouseTracks(self):
        import umap
        import matplotlib.patches as mpatches
        from sklearn.cluster import AgglomerativeClustering

        sceneInfo = self.sceneInfo

        FristHalfPos = []
        fullTrackFlat = []
        IndexEveryday = self.dayInstance.trialIndexes
        Pos = self.dayInstance.ReturnPosPerTrial()
        TimeMark = self.dayInstance.ReturnTrialTimeMark()
        # TrialDirection.append(np.asarray(, dtype= int))
        for track in range(len(Pos)):
            fullTrackFlat.append(Pos[track][['smooth_x', 'smooth_y']].values)
            fhInd = TimeMark[track][2]
            FristHalfPos.append(Pos[track].loc[0:fhInd][['smooth_x', 'smooth_y']].values)

        distancesFlat = self.dayInstance.ReturnDistancePerTrial()
        fhDistances = self.dayInstance.ReturnFirstHalfDistancePerTrial()
        _shortInd = np.arange(len(distancesFlat))[np.array(distancesFlat) < sceneInfo[2] * 4]
        # _longInd  = np.array(range(len(distancesFlat)))[distancesFlat >= sceneInfo[2] * 4]
        fhTrackFlat = FristHalfPos
        indexesFlat = self.dayInstance.trialIndexes
        TrialDirFlat = self.dayInstance.licksPos

        self.ResampleClusteredData['distanceFlat']  = distancesFlat
        self.ResampleClusteredData['fhDistances'] = fhDistances
        self.ResampleClusteredData['fhTrackFlat']   = fhTrackFlat
        self.ResampleClusteredData['fullTrackFlat'] = fullTrackFlat
        self.ResampleClusteredData['indexesFlat']   = indexesFlat
        self.ResampleClusteredData['TrialDirFlat']  = TrialDirFlat

        ResampledfullTrack = []
        ResampledFullTrackRotated = []
        for i, track in enumerate(fullTrackFlat):
            ResampledfullTrack.append(ResampleTrack(track))
            ResampledFullTrackRotated.append(rotate_points(ResampledfullTrack[-1], sceneInfo[0], sceneInfo[1], - (TrialDirFlat[i])))
        fhFlatShort = []
        fhFlatLong = []
        # segTrackflatShort = []
        segTrackflatLong = []
        dirFlatShort = []
        dirFlatLong = []
        idx_pathsShort = []
        idx_pathsLong = []
        indexFlatShort = []
        indexFlatLong = []
        # marksFlatShort = []
        # marksFlatLong = []
        # fullTrajFlatShort = []
        # fullTrajFlatLong = []
        ResampledShortTrack = []
        ResampledShortfullTrack = []
        ResampledShortTrackRotated = []
        ResampledShortfullTrackRotated = []
        for i in range(len(indexesFlat)):
            if i in _shortInd:
                indexFlatShort.append(indexesFlat[i])
                fhFlatShort.append(fhTrackFlat[i])
                dirFlatShort.append(TrialDirFlat[i])
                # ResampledShortTrack.append(ResampleTrack(fullTrackFlat[i].T))
                ResampledShortTrack.append(ResampleTrack(fhTrackFlat[i].T))
                ResampledShortfullTrack.append(ResampleTrack(fullTrackFlat[i].T))
                ResampledShortTrackRotated.append(rotate_points(ResampledShortTrack[-1], sceneInfo[0], sceneInfo[1], - dirFlatShort[-1]))
                ResampledShortfullTrackRotated.append(rotate_points(ResampledShortfullTrack[-1], sceneInfo[0], sceneInfo[1], - dirFlatShort[-1]))
            else:
                idx_pathsLong.append(indexesFlat[i])
                fhFlatLong.append(fhTrackFlat[i])
                dirFlatLong.append(TrialDirFlat[i])

        # testTracks, testLabels = generate_synthetic_trajectories()
        _temprst = np.array(ResampledFullTrackRotated, dtype= float)
        testTracks = np.concatenate([_temprst[:, :, 0], _temprst[:, :, 1]], axis = 1)

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(testTracks)
        nClsC = 2
        clustering = AgglomerativeClustering(
            n_clusters=nClsC, 
            metric='euclidean',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(embedding)

        self.ResampleClusteredData['cluster_labels'] = cluster_labels

        _fig:plt.Figure = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=cluster_labels, 
            cmap='jet',
            s=50,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        cmap = plt.get_cmap('jet', nClsC)
        handles = []
        for i in range(nClsC):
            color = cmap(i)  # 直接通过索引获取颜色
            handles.append(mpatches.Patch(color=color, label=str(i)))

        # 添加图例
        plt.legend(handles=handles)
        _fig.show()

        _fig, _axes = plt.subplots(2, 5, figsize = (20, 5), dpi = 300)
        # norm = plt.Normalize(0, 1, clip=True)
        import colorsys

        colors = generate_rainbow_colors(99)

        # _axes[-1].scatter(range(200, 1190, 10), [900]*99, c = colors, alpha = 0.5, s = 5)
        for i in range(nClsC):
            tracks = _temprst[np.array(cluster_labels) == i]
            for _it, track in enumerate(tracks):
                _track = track.copy()
                _track[:, 1] = 1080 - _track[:, 1]
                _track = _track.reshape(-1, 1, 2)
                segments_vis = np.concatenate([_track[:-1], _track[1:]], axis=1)
                lc = LineCollection(segments_vis, colors = colors, linewidth=0.1, alpha = 0.1)
                _axes[0][i].add_collection(lc)

            _tracks = np.array(tracks)
            _averagedTrack = np.mean(_tracks, axis=0)
            _averagedTrack[:, 1] = 1080 - _averagedTrack[:, 1]
            _averagedTrack = _averagedTrack.reshape(-1, 1, 2)
            segments_vis = np.concatenate([_averagedTrack[:-1],  _averagedTrack[1:]], axis=1)
            lc = LineCollection(segments_vis, colors = colors, linewidth=2, alpha = 1)
            _axes[1][i].add_collection(lc)
            _axes[1][i].set_xlim(0, 1440)
            _axes[1][i].set_ylim(0, 1080)
            _axes[1][i].set_xticks([])
            _axes[1][i].set_yticks([])
            _axes[1][i].set_aspect('equal')
            circles:list[plt.Circle] = [
                plt.Circle((sceneInfo[0], 1080 - sceneInfo[1]), sceneInfo[2], color='black', fill=False, linewidth = 1),
                plt.Circle((sceneInfo[0] + 372, 1080 - sceneInfo[1]), 120, color='r', fill=False, linewidth=1),
            ]
            _circles:list[plt.Circle] = [
                plt.Circle((sceneInfo[0], 1080 - sceneInfo[1]), sceneInfo[2], color='black', fill=False, linewidth = 1),
                plt.Circle((sceneInfo[0] + 372, 1080 - sceneInfo[1]), 120, color='r', fill=False, linewidth=1),
            ]
            for circle in circles:
                _axes[0][i].add_patch(circle)
            for circle in _circles:
                _axes[1][i].add_patch(circle)
                # _axes[i].plot(track[:, 0], track[:, 1], lw = 0.1, alpha = 0.01, c= DailyColor[i * 4])
            # _axes[i].scatter([sceneInfo[0]], [1080 - sceneInfo[1]], c = 'black', s=sceneInfo[2] * 2, marker = 'o')
            # _axes[i].scatter([sceneInfo[0] + 372], [1080 - sceneInfo[1]], c = 'red', s=120 * 2, marker = 'o')
            _axes[0][i].annotate(f"Cluster {i} count {len(cluster_labels[cluster_labels == i])}", (100, 80), fontsize=12)
            _axes[0][i].set_xlim(0, 1440)
            _axes[0][i].set_ylim(0, 1080)
            _axes[0][i].set_xticks([])
            _axes[0][i].set_yticks([])
            _axes[0][i].set_aspect('equal')
        # _fig.show()
        plt.show()

    def GetActiveNeuronIndexInSelectedTime(self, _timeStart:np.ndarray, _timeEnd:np.ndarray, participateThreshold:float, peakThreshold:float, ignoreMultipleFire:bool = True, useDeconvolved:bool = False) -> np.ndarray:
        assert len(_timeStart) == len(_timeEnd)
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData

        peakThreshold:np.ndarray = np.percentile(neuronData, peakThreshold * 100, axis=1, keepdims=True)
        trialFracStage = self.RelatedBehaviorData['tempFracStage']
        selectedIndexes, activity = self.GetNeuronActivityInSelectedTime(_timeStart, _timeEnd)
        tempFracStage = [np.mean(trialFracStage[ind]) for ind in selectedIndexes]
        activity = activity > peakThreshold
        participateStatus = [np.array([sum(peakPerNeuron) / len(peakPerNeuron) if not ignoreMultipleFire else min(1, sum(peakPerNeuron)) for peakPerNeuron in activity])]
        # selectedIndexes = np.concatenate(selectedIndexes)
        # for i in range(len(_timeStart)):
        #     selectedMSFrame = self.MSFrameTimeSegmented[((self.MSFrameTimeSegmented > _timeStart[i]) & (self.MSFrameTimeSegmented < _timeEnd[i]))]
        #     if len(selectedMSFrame):
        #         neuronDataIndexInSelectedTime = self.GetCorrespondNeuronDataIndex(selectedMSFrame)
        #         peaks = self.neuronData[:, neuronDataIndexInSelectedTime] >= peakThreshold
        #         # peaks = Deconvolve(peaks, 20)#用deconvoloed结果
        #         tempFracStage.append(np.mean(trialFracStage[neuronDataIndexInSelectedTime]))
        #         participateStatus.append(np.array([sum(peakPerNeuron) / len(peakPerNeuron) if not ignoreMultipleFire else min(1, sum(peakPerNeuron)) for peakPerNeuron in peaks]))
        participateStatusSum = np.array(participateStatus).sum(axis=0)
        activeNeuron = np.where(participateStatusSum >= len(participateStatus) * participateThreshold)
        if len(activeNeuron[0]) == 0:
            plt.hist(participateStatusSum / len(participateStatus), bins=50)
        plt.hist(tempFracStage, bins= 20)
        plt.show()
        return self.neuronIndex[activeNeuron]
    
    def GetActiveNeuronIndexInSelectedRegion(self, pos:Tuple, radius:float, participateThreshold:float, peakThreshold:float, concentrationThreshold:float = 0, useDeconvoloved:bool = False):
        assert len(pos) == 2

        peakIndexes = self.MSFrameInPosDfIndexExtracted
        relatedPos = self.GetRelatedPos(peakIndexes)
        timeMark = np.vstack(self.dayInstance.ReturnTrialTimeMarkCollected())[:, 1]
        timeMark = self.GetCorrespondNeuronDataIndex(timeMark)
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData

        distances = np.sqrt(np.array((relatedPos[:, 0] - pos[0])**2 + (relatedPos[:, 1] - pos[1])**2, dtype = float))
        inRegionMask = distances <= radius
        totalRegionFrames = np.sum(inRegionMask)

        if totalRegionFrames == 0:
            return np.array([], dtype=int)

        activity_thresholds = np.percentile(neuronData, peakThreshold * 100, axis=1)
        is_active = neuronData > activity_thresholds[:, np.newaxis]
        # total_activation_events = np.sum(is_active, axis=1)

        activationInRegion = np.sum(is_active & inRegionMask, axis=1)
        activationInRegionSplitByTrial = np.array([np.sum(is_active[:, timeMark[i]:timeMark[i+1]], axis = 1) for i in range(len(timeMark)-1)]).T

        # participationRatios = activationInRegion / totalRegionFrames
        participationRatiosSplitByTrial = np.sum(activationInRegionSplitByTrial > 0, axis = 1) / activationInRegionSplitByTrial.shape[1]

        # concentration_ratios = np.divide(
        #     activationInRegion, 
        #     total_activation_events, 
        #     where=total_activation_events > 0
        # )
        normalizedActivity = neuronData / activity_thresholds[:, np.newaxis]
        normalizedActivity = normalizedActivity - np.min(normalizedActivity, axis=1)[:, np.newaxis]
        normalizedActivity = normalizedActivity / np.max(normalizedActivity, axis=1)[:, np.newaxis]
        concentration_ratios = np.sum(normalizedActivity[:, inRegionMask], axis = 1) /np.sum(normalizedActivity, axis = 1)
        
        # participate_mask = participationRatios >= participateThreshold
        participate_mask = participationRatiosSplitByTrial >= participateThreshold
        concentration_mask = concentration_ratios >= concentrationThreshold
        valid_neurons_mask = participate_mask & concentration_mask

        print("participation_ratios(blue) and concentration_ratios(yellow) as below")
        plt.hist(participationRatiosSplitByTrial, bins=50, color = 'b', alpha = 0.5)
        plt.hist(concentration_ratios, bins=50, color = 'y', alpha = 0.5)
        plt.show()
        # 返回满足条件的神经元索引
        return self.neuronIndex[valid_neurons_mask]
    
    def GetNeuronActivityInSelectedTime(self, _timeStart:np.ndarray, _timeEnd:np.ndarray, useDeconvolved:bool = False) -> Tuple[List[int], np.ndarray]:
        '''返回：index（长度等于NeuronData），Acitivity'''
        assert len(_timeStart) == len(_timeEnd)
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData

        neuronDataIndexesInSelectedTime = []
        for i in range(len(_timeStart)):
            selectedMSFrame = self.MSFrameTimeSegmented[((self.MSFrameTimeSegmented > _timeStart[i]) & (self.MSFrameTimeSegmented < _timeEnd[i]))]
            if len(selectedMSFrame):
                neuronDataIndexInSelectedTime = self.GetCorrespondNeuronDataIndex(selectedMSFrame)
                neuronDataIndexesInSelectedTime.append(neuronDataIndexInSelectedTime)
                # peaks = self.neuronData[:, neuronDataIndexInSelectedTime] >= peakThreshold
        selectedIndexes = neuronDataIndexesInSelectedTime
        neuronDataIndexesInSelectedTime = np.concatenate(neuronDataIndexesInSelectedTime)
        activity = neuronData[:, neuronDataIndexesInSelectedTime]
        return selectedIndexes, activity

    def ShowNeuronActivityAlignWithOffset(self, variable:List, splitRatio:float = 0.6, offset:float = 0, useDeconvolved:bool = False):
        def GetOffsetedData(arr, offset, fill_value = None):
            if fill_value is None:
                return np.concatenate((
                        np.full(offset, arr[0]) if offset > 0 else np.array([]),
                        arr[max(-offset, 0):min(len(arr)-offset, len(arr))],
                        np.full(-offset, arr[-1]) if offset < 0 else np.array([])
                    ))
            else:
                return np.concatenate((
                            np.full(offset, fill_value) if offset > 0 else np.array([]),
                            arr[max(-offset, 0):min(len(arr)-offset, len(arr))],
                            np.full(-offset, fill_value) if offset < 0 else np.array([])
                        ))
        offset = int(offset / 0.1)
        fill_value = None
        cluster_labels      = GetOffsetedData(self.ResampleClusteredData['cluster_labels'], offset, fill_value)
        indexes             = GetOffsetedData(self.ResampleClusteredData['indexesFlat'], offset, fill_value)
        tempFracStage       = GetOffsetedData(self.RelatedBehaviorData['tempFracStage'], offset, fill_value)
        tempSpeed           = GetOffsetedData(self.RelatedBehaviorData['tempSpeed'], offset, fill_value)
        angleData           = GetOffsetedData(self.RelatedBehaviorData['angleData'], offset, fill_value)
        angleDeltaData      = GetOffsetedData(self.RelatedBehaviorData['angleDeltaData'], offset, fill_value)
        stageRelatedTrial   = GetOffsetedData(np.array(self.RelatedBehaviorData['tempStage'], dtype= int), offset, fill_value)
        randomSelectedTrials = np.random.choice(np.unique(stageRelatedTrial), int(len(np.unique(stageRelatedTrial)) * splitRatio), replace=False)
        relatedTrialCorrCluster = []
        for i in range(len(cluster_labels)):
            relatedTrialCorrCluster.append([indexes[i], cluster_labels[i]])
        maxCount = max(np.array(relatedTrialCorrCluster)[:,0])
        for i in np.setdiff1d(range(maxCount+10), np.array(relatedTrialCorrCluster)[:,0]):
            relatedTrialCorrCluster.append([i, -1])
        relatedTrialCorrCluster = np.array(relatedTrialCorrCluster)
        relatedTrialCorrCluster = relatedTrialCorrCluster[relatedTrialCorrCluster[:,0].argsort()]

        mask = relatedTrialCorrCluster[:,1] != -1
        relatedTrialCorrCluster = relatedTrialCorrCluster[mask, :]
        import seaborn as sns
        from scipy.stats import pearsonr

        clusteredTrials = []
        clusterMasks = []
        try:
            for i in np.unique(relatedTrialCorrCluster[:, 1]):
                clusteredTrials.append(relatedTrialCorrCluster[relatedTrialCorrCluster[:, 1] == i][:, 0])
                clusterMasks.append(np.array([True if tr in clusteredTrials[-1] else False for tr in stageRelatedTrial]))
        except:
            pass

        # clusterMasks = [np.array([True] * len(stageRelatedTrial))]
        neuronData = self.neuronDataDeconvolved if useDeconvolved else self.neuronData

        for ClusterMask in clusterMasks:
            ClusteredNeuronData = neuronData[:, ClusterMask]
            tempFracStageChunked = np.array(tempFracStage[ClusterMask] / 0.05, dtype= int) * 0.05
            if angleData is not None:
                tempHeadDirChunked = np.array((angleData[ClusterMask][:, 2] + 180) /10, dtype= int) *10
                tempHeadDirDeltaChunked = np.array(angleDeltaData[ClusterMask][:, 2] / 10, dtype= int) * 10
            tempSpeedChunked = np.array(tempSpeed[ClusterMask] / 10, dtype= int) * 10
            neuronFracStageData = []
            neuronFracStageValidateData = []
            neuronHeadDirData = []
            neuronHeadDirValidateData = []
            neuronHeadDirDeltaData = []
            neuronHeadDirDeltaValidateData = []
            neuronSpeedData = []
            neuronSpeedValidateData = []
            neuronClusterData = []
            neuronClusterValidateData = []
            selectedTrialRelatedDataMask = np.array([True if srt in randomSelectedTrials else False for srt in stageRelatedTrial])
            selectedTrialRelatedDataMask = selectedTrialRelatedDataMask[ClusterMask]
            selectedTrialRelatedDataRowCount = np.sum(selectedTrialRelatedDataMask)
            for i in range(20):
                mask = np.array([tempFracStageChunked == i * 0.05]).reshape(np.sum(ClusterMask))
                unselectMask = np.array(~selectedTrialRelatedDataMask & mask).reshape(len(mask))
                mask = np.array(selectedTrialRelatedDataMask & mask).reshape(len(mask))
                neuronFracStageData.append(np.sum(ClusteredNeuronData[:, mask], axis = 1) / np.sum(mask))
                neuronFracStageValidateData.append(np.sum(ClusteredNeuronData[:, unselectMask], axis = 1) / np.sum(unselectMask))
            neuronFracStageData = np.array(neuronFracStageData, dtype= float)
            neuronFracStageValidateData = np.array(neuronFracStageValidateData, dtype= float)
            
            if angleData is not None:
                for i in range(36):
                    # neuronHeadDirData.append(np.sum(neurondata[:, tempHeadDirChunked == i * 10], axis = 1) / len(tempHeadDirChunked[tempHeadDirChunked == i * 10]))
                    # neuronHeadDirData.append(np.sum(neurondata[:, tempHeadDirChunked == i * 10], axis = 1))
                    mask = (np.array([tempHeadDirChunked == i * 10]) & [tempFracStage[ClusterMask] > 0.1]).reshape(np.sum(ClusterMask))
                    unselectMask = np.array(~selectedTrialRelatedDataMask & mask).reshape(len(mask))
                    mask = np.array(selectedTrialRelatedDataMask & mask).reshape(len(mask))
                    neuronHeadDirData.append(np.sum(ClusteredNeuronData[:, mask], axis = 1) / np.sum(mask))
                    neuronHeadDirValidateData.append(np.sum(ClusteredNeuronData[:, unselectMask], axis = 1) / np.sum(unselectMask))
                    # neuronHeadDirData.append(np.sum(neurondata[:, mask], axis = 1))
                neuronHeadDirData = np.array(neuronHeadDirData, dtype= float)
                neuronHeadDirValidateData = np.array(neuronHeadDirValidateData, dtype= float)

                neuronHeadDirDeltaDataLabels = []
                for i in range(-18, 18):
                    if np.sum([tempHeadDirDeltaChunked == i * 10]) > 20:
                    # neuronHeadDirData.append(np.sum(neurondata[:, tempHeadDirChunked == i * 10], axis = 1) / len(tempHeadDirChunked[tempHeadDirChunked == i * 10]))
                        mask = (np.array([tempHeadDirDeltaChunked == i * 10]) & [tempFracStage[ClusterMask] > 0.1]).reshape(np.sum(ClusterMask))
                        unselectMask = np.array(~selectedTrialRelatedDataMask & mask).reshape(len(mask))
                        mask = np.array(selectedTrialRelatedDataMask & mask).reshape(len(mask))
                        neuronHeadDirDeltaData.append(np.sum(ClusteredNeuronData[:, mask], axis = 1) / np.sum(mask))
                        neuronHeadDirDeltaValidateData.append(np.sum(ClusteredNeuronData[:, unselectMask], axis = 1) / np.sum(unselectMask))
                        # neuronHeadDirDeltaData.append(np.sum(neurondata[:, mask], axis = 1) / len(tempHeadDirDeltaChunked[tempHeadDirDeltaChunked == i * 10]))
                        neuronHeadDirDeltaDataLabels.append(str(i * 10))
                neuronHeadDirDeltaData = np.array(neuronHeadDirDeltaData, dtype= float)
                neuronHeadDirDeltaValidateData = np.array(neuronHeadDirDeltaValidateData, dtype= float)

            for i in range(int(max(tempSpeed) / 10) + 1):
                if np.sum([tempSpeedChunked == i * 10]) > 20:
                # neuronHeadDirData.append(np.sum(neurondata[:, tempHeadDirChunked == i * 10], axis = 1) / len(tempHeadDirChunked[tempHeadDirChunked == i * 10]))
                    mask = (np.array([tempSpeedChunked == i * 10]) & [tempFracStage[ClusterMask] > 0.1]).reshape(np.sum(ClusterMask))
                    unselectMask = np.array(~selectedTrialRelatedDataMask & mask).reshape(len(mask))
                    mask = np.array(selectedTrialRelatedDataMask & mask).reshape(len(mask))
                    neuronSpeedData.append(np.sum(ClusteredNeuronData[:, mask], axis = 1) / np.sum(mask))
                    neuronSpeedValidateData.append(np.sum(ClusteredNeuronData[:, unselectMask], axis = 1) / np.sum(unselectMask))
                    # neuronSpeedData.append(np.sum(neurondata[:, mask], axis = 1) / len(tempSpeedChunked[tempSpeedChunked == i * 10]))
            neuronSpeedData = np.array(neuronSpeedData, dtype= float)
            neuronSpeedValidateData = np.array(neuronSpeedValidateData, dtype= float)


            neuronFracStageDataNormalized = neuronFracStageData.T
            neuronFracStageDataNormalized = (neuronFracStageDataNormalized - neuronFracStageDataNormalized.min(axis=1, keepdims=True)) / (neuronFracStageDataNormalized.max(axis=1, keepdims=True) - neuronFracStageDataNormalized.min(axis=1, keepdims=True))
            neuronFracStageDataDf = pd.DataFrame(neuronFracStageDataNormalized)
            neuronFracStageDataDf['max'] = np.argmax(neuronFracStageDataNormalized, axis=1)
            neuronFracStageDataDf = neuronFracStageDataDf.sort_values(by='max')
            neuronFracStageDataDf = neuronFracStageDataDf.drop('max', axis=1)

            neuronFracStageValidateDataNormalized = neuronFracStageValidateData.T
            neuronFracStageValidateDataNormalized = (neuronFracStageValidateDataNormalized - neuronFracStageValidateDataNormalized.min(axis=1, keepdims=True)) / (neuronFracStageValidateDataNormalized.max(axis=1, keepdims=True) - neuronFracStageValidateDataNormalized.min(axis=1, keepdims=True))
            neuronFracStageValidateDataDf = pd.DataFrame(neuronFracStageValidateDataNormalized)
            # neuronFracStageValidateDataDf['max'] = np.argmax(neuronFracStageValidateDataNormalized, axis=1)
            neuronFracStageValidateDataDf = neuronFracStageValidateDataDf.loc[neuronFracStageDataDf.index]
            # neuronFracStageValidateDataDf = neuronFracStageValidateDataDf.drop('max', axis=1)

            if angleData is not None:
                neuronHeadDirDataNormalized = neuronHeadDirData.T
                neuronHeadDirDataNormalized = (neuronHeadDirDataNormalized - neuronHeadDirDataNormalized.min(axis=1, keepdims=True)) / (neuronHeadDirDataNormalized.max(axis=1, keepdims=True) - neuronHeadDirDataNormalized.min(axis=1, keepdims=True))
                neuronHeadDirDataDf = pd.DataFrame(neuronHeadDirDataNormalized)
                neuronHeadDirDataDf['max'] = np.argmax(neuronHeadDirDataNormalized, axis=1)
                neuronHeadDirDataDf = neuronHeadDirDataDf.sort_values(by='max')
                neuronHeadDirDataDf = neuronHeadDirDataDf.drop('max', axis=1)

                neuronHeadDirValidateDataNormalized = neuronHeadDirValidateData.T
                neuronHeadDirValidateDataNormalized = (neuronHeadDirValidateDataNormalized - neuronHeadDirValidateDataNormalized.min(axis=1, keepdims=True)) / (neuronHeadDirValidateDataNormalized.max(axis=1, keepdims=True) - neuronHeadDirValidateDataNormalized.min(axis=1, keepdims=True))
                neuronHeadDirValidateDataDf = pd.DataFrame(neuronHeadDirValidateDataNormalized)
                neuronHeadDirValidateDataDf = neuronHeadDirValidateDataDf.loc[neuronHeadDirDataDf.index]


                neuronHeadDirDeltaDataNormalized = neuronHeadDirDeltaData.T
                neuronHeadDirDeltaDataNormalized = (neuronHeadDirDeltaDataNormalized - neuronHeadDirDeltaDataNormalized.min(axis=1, keepdims=True)) / (neuronHeadDirDeltaDataNormalized.max(axis=1, keepdims=True) - neuronHeadDirDeltaDataNormalized.min(axis=1, keepdims=True))
                neuronHeadDirDeltaDataDf = pd.DataFrame(neuronHeadDirDeltaDataNormalized)
                neuronHeadDirDeltaDataDf['max'] = np.argmax(neuronHeadDirDeltaDataNormalized, axis=1)
                neuronHeadDirDeltaDataDf = neuronHeadDirDeltaDataDf.sort_values(by='max')
                neuronHeadDirDeltaDataDf = neuronHeadDirDeltaDataDf.drop('max', axis=1)

                neuronHeadDirDeltaValidateDataNormalized = neuronHeadDirDeltaValidateData.T
                neuronHeadDirDeltaValidateDataNormalized = (neuronHeadDirDeltaValidateDataNormalized - neuronHeadDirDeltaValidateDataNormalized.min(axis=1, keepdims=True)) / (neuronHeadDirDeltaValidateDataNormalized.max(axis=1, keepdims=True) - neuronHeadDirDeltaValidateDataNormalized.min(axis=1, keepdims=True))
                neuronHeadDirDeltaValidateDataDf = pd.DataFrame(neuronHeadDirDeltaValidateDataNormalized)
                neuronHeadDirDeltaValidateDataDf = neuronHeadDirDeltaValidateDataDf.loc[neuronHeadDirDeltaDataDf.index]

            neuronSpeedDataNormalized = neuronSpeedData.T
            neuronSpeedDataNormalized = (neuronSpeedDataNormalized - neuronSpeedDataNormalized.min(axis=1, keepdims=True)) / (neuronSpeedDataNormalized.max(axis=1, keepdims=True) - neuronSpeedDataNormalized.min(axis=1, keepdims=True))
            neuronSpeedDf = pd.DataFrame(neuronSpeedDataNormalized)
            neuronSpeedDf['max'] = np.argmax(neuronSpeedDataNormalized, axis=1)
            neuronSpeedDf = neuronSpeedDf.sort_values(by='max')
            neuronSpeedDf = neuronSpeedDf.drop('max', axis=1)

            neuronSpeedValidateDataNormalized = neuronSpeedValidateData.T
            neuronSpeedValidateDataNormalized = (neuronSpeedValidateDataNormalized - neuronSpeedValidateDataNormalized.min(axis=1, keepdims=True)) / (neuronSpeedValidateDataNormalized.max(axis=1, keepdims=True) - neuronSpeedValidateDataNormalized.min(axis=1, keepdims=True))
            neuronSpeedValidateDataDf = pd.DataFrame(neuronSpeedValidateDataNormalized)
            neuronSpeedValidateDataDf = neuronSpeedValidateDataDf.loc[neuronSpeedDf.index]

            # neuronClusterDataNormalized = neuronClusterData.T
            # neuronClusterDataNormalized = (neuronClusterDataNormalized - neuronClusterDataNormalized.min(axis=1, keepdims=True)) / (neuronClusterDataNormalized.max(axis=1, keepdims=True) - neuronClusterDataNormalized.min(axis=1, keepdims=True))
            # neuronClusterDataDf = pd.DataFrame(neuronClusterDataNormalized)
            # neuronClusterDataDf['max'] = np.argmax(neuronClusterDataNormalized, axis=1)
            # neuronClusterDataDf = neuronClusterDataDf.sort_values(by='max')
            # neuronClusterDataDf = neuronClusterDataDf.drop('max', axis=1)

            # neuronClusterValidateDataNormalized = neuronClusterValidateData.T
            # neuronClusterValidateDataNormalized = (neuronClusterValidateDataNormalized - neuronClusterValidateDataNormalized.min(axis=1, keepdims=True)) / (neuronClusterValidateDataNormalized.max(axis=1, keepdims=True) - neuronClusterValidateDataNormalized.min(axis=1, keepdims=True))
            # neuronClusterValidateDataDf = pd.DataFrame(neuronClusterValidateDataNormalized)
            # neuronClusterValidateDataDf = neuronClusterValidateDataDf.loc[neuronClusterDataDf.index]

            # sns.heatmap(neuronFracStageDataDf)
            LastNeuronIndex = len(neuronData) - 1
            _fig, _axes = plt.subplots(2, 4, figsize=(20, 10), dpi = 200)
            print(f'now data rows: {np.sum(ClusterMask)}')
            # print(f'taskStage corrcoef: {np.corrcoef(neuronFracStageDataNormalized.flatten(), neuronFracStageValidateDataNormalized.flatten())}')
            print(f'taskStage pearson correlation coefficient: {pearsonr(neuronFracStageDataNormalized.flatten(), neuronFracStageValidateDataNormalized.flatten())}')
            cmap='coolwarm'
            tempAxe:plt.Axes = sns.heatmap(neuronFracStageDataDf, ax = _axes[0][0], xticklabels = [f'{i * 0.05:.2f}' for i in range(0, 20)], cmap=cmap)
            tempAxe.set_xticks(range(0, 22, 2), [f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)])
            tempAxe.set_yticks([0, LastNeuronIndex], ['0', f'{LastNeuronIndex}'])
            _axes[0][0].set_title('Task Stage', size = 14, weight = 'bold')
            tempAxe:plt.Axes = sns.heatmap(neuronFracStageValidateDataDf, ax = _axes[1][0], xticklabels = [f'{i * 0.05:.2f}' for i in range(0, 20)], cmap=cmap)
            tempAxe.set_xticks(range(0, 22, 2), [f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)])
            # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])
            # _axes[1][0].set_title('Task Stage')
            # print(f'HeadDir corrcoef: {np.corrcoef(neuronHeadDirDataNormalized.flatten(), neuronHeadDirValidateDataNormalized.flatten())}')
            
            if angleData is not None:
                print(f'HeadDir pearson correlation coefficient: {pearsonr(neuronHeadDirDataNormalized.flatten(), neuronHeadDirValidateDataNormalized.flatten())}')
                tempAxe:plt.Axes = sns.heatmap(neuronHeadDirDataDf, ax = _axes[0][1], xticklabels = [str(i * 20) for i in range(0, 36)], cmap=cmap)
                tempAxe.set_xticks(range(0, 37, 2), [f"{i}" for i in np.arange(0, 361, 20)])
                tempAxe.set_yticks([0, LastNeuronIndex], ['0', f'{LastNeuronIndex}'])
                _axes[0][1].set_title('Absolute Head Direction', size = 14, weight = 'bold')
                tempAxe:plt.Axes = sns.heatmap(neuronHeadDirValidateDataDf, ax = _axes[1][1], xticklabels = [str(i * 200) for i in range(0, 36)], cmap=cmap)
                # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])
                tempAxe.set_xticks(range(0, 37, 2), [f"{i}" for i in np.arange(0, 361, 20)])
                # _axes[1][1].set_title('Absolute Head Direction')
                # print(f'HeadDirDelta corrcoef: {np.corrcoef(neuronHeadDirDeltaDataNormalized.flatten(), neuronHeadDirDeltaValidateDataNormalized.flatten())}')
                print(f'HeadDirDelta pearson correlation coefficient: {pearsonr(neuronHeadDirDeltaDataNormalized.flatten(), neuronHeadDirDeltaValidateDataNormalized.flatten())}')
                tempAxe:plt.Axes = sns.heatmap(neuronHeadDirDeltaDataDf, ax = _axes[0][2], xticklabels = neuronHeadDirDeltaDataLabels, cmap=cmap)
                tempAxe.set_yticks([0, LastNeuronIndex], ['0', f'{LastNeuronIndex}'])
                # tempAxe.set_xticks(range(12), [f"{i}" for i in np.arange(-50, 61, 10)])
                _axes[0][2].set_title('Angular Head Velocity', size = 14, weight = 'bold')
                tempAxe:plt.Axes = sns.heatmap(neuronHeadDirDeltaValidateDataDf, ax = _axes[1][2], xticklabels = neuronHeadDirDeltaDataLabels, cmap=cmap)
                # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])
                # tempAxe.set_xticks(range(12), [f"{i}" for i in np.arange(-50, 61, 10)])
                # _axes[0][2].set_title('Angular Head Velocity')
                # print(f'Speed corrcoef: {np.corrcoef(neuronSpeedDataNormalized.flatten(), neuronSpeedValidateDataNormalized.flatten())}')
            print(f'Speed pearson correlation coefficient: {pearsonr(neuronSpeedDataNormalized.flatten(), neuronSpeedValidateDataNormalized.flatten())}')
            tempAxe:plt.Axes = sns.heatmap(neuronSpeedDf, ax = _axes[0][3], cmap=cmap)
            tempAxe.set_yticks([0, LastNeuronIndex], ['0', f'{LastNeuronIndex}'])
            _axes[0][3].set_title('Speed', size = 14, weight = 'bold')
            tempAxe:plt.Axes = sns.heatmap(neuronSpeedValidateDataDf, ax = _axes[1][3], cmap=cmap)
            # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])
            plt.show()
            # _axes[1][3].set_title('Speed')
            # tempAxe:plt.Axes = sns.heatmap(neuronClusterDataDf, ax = _axes[0][4], cmap=cmap)
            # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])
            # _axes[0][4].set_title('Cluster', size = 14, weight = 'bold')
            # tempAxe:plt.Axes = sns.heatmap(neuronClusterValidateDataDf, ax = _axes[1][4], cmap=cmap)
            # tempAxe.set_yticks([0, neuronCount], ['0', f'{neuronCount}'])

            # _axes[1][4].set_title('Cluster')
            # _axes[2].set_xlabel(neuronHeadDirDeltaDataLabels)

def _process_single_offset(
        offset: float,
        neuron_activity: np.ndarray,
        chunknum: int,
        get_pos_index_func,     # 函数对象：GetCorrespondPosDfIndexWithOffset
        get_related_pos_func,   # 函数对象：GetRelatedPos
        transformed_x: float,
        transformed_y: float,
        chunkScale: float,
        position_prior_map: np.ndarray
    ) -> Dict[str, Any]:
        """
        独立处理单个 offset，无类依赖，适合并行。
        """
        try:
            # 获取位置索引和坐标
            peakIndexes = get_pos_index_func(offset)
            relatedPos = get_related_pos_func(peakIndexes)

            # 计算网格坐标
            chunkedposdata = np.array([
                (relatedPos[:, 0] - transformed_x) / chunkScale,
                (relatedPos[:, 1] - transformed_y) / chunkScale
            ], dtype=int).T

            chunkedposdata = np.clip(chunkedposdata, 0, chunknum - 1)

            # 构建 index_map
            index_map = {}
            for idx, coord in enumerate(chunkedposdata):
                key = (chunknum - 1 - coord[1]) * chunknum + coord[0]
                if key not in index_map:
                    index_map[key] = []
                index_map[key].append(idx)

            # 预处理神经活动（移到外部已做，这里复用）
            peak = neuron_activity / np.percentile(neuron_activity, 99)
            peak = np.clip(peak, 0, 1) ** 2
            p_min, p_max = np.min(peak), np.max(peak)
            peak = (peak - p_min) / (p_max - p_min)


            # 计算基础兴奋野
            clustedActivityData = np.zeros((chunknum ** 2))
            for k in index_map:
                activity = peak[index_map[k]]
                clustedActivityData[k] = np.sum(activity) / np.sqrt(len(index_map[k]))

            # Bootstrap 抽样评估稳定性
            bootstrap_maps = []
            for _ in range(50):
                resampled_indices = resample(np.arange(len(peak)))
                resampled_peak = peak[resampled_indices]
                resampled_chunkedposdata = chunkedposdata[resampled_indices]

                resampled_index_map = {}
                for idx, coord in enumerate(resampled_chunkedposdata):
                    key = (chunknum - 1 - coord[1]) * chunknum + coord[0]
                    if key not in resampled_index_map:
                        resampled_index_map[key] = []
                    resampled_index_map[key].append(idx)

                resampled_activity_data = np.zeros((chunknum ** 2))
                for k in resampled_index_map:
                    activity = resampled_peak[resampled_index_map[k]]
                    resampled_activity_data[k] = np.sum(activity) / np.sqrt(len(resampled_index_map[k]))

                bootstrap_maps.append(resampled_activity_data)

            # 计算稳定性：bootstrap 地图两两相关性均值
            stability = 0
            count = 0
            n_boot = len(bootstrap_maps)
            for i in range(n_boot):
                for j in range(i + 1, n_boot):
                    valid_mask = (bootstrap_maps[i] > 0) | (bootstrap_maps[j] > 0)
                    if np.sum(valid_mask) > 10:
                        corr, _ = pearsonr(bootstrap_maps[i][valid_mask], bootstrap_maps[j][valid_mask])
                        stability += corr
                        count += 1

            stability = stability / count if count > 0 else 0

            # 计算集中度
            smoothed_map = gaussian_filter(clustedActivityData.reshape(chunknum, chunknum), sigma=1)
                # 1. 峰值占比
            max_activity = np.max(smoothed_map)
            total_activity = np.sum(smoothed_map)
            peak_ratio = max_activity / total_activity

            # 2. 空间熵（衡量分布集中性）
            p = smoothed_map / total_activity
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(smoothed_map.size)
            spatial_focus = 1.0 - (entropy / max_entropy)  # 越接近1越集中

            # 3. 有效格子数惩罚（防止单点爆发）
            active_mask = smoothed_map > (max_activity * 0.1)
            active_bins = np.sum(active_mask)
            min_bins = 3
            bin_penalty = min(1.0, active_bins / min_bins)

            # 4. 孤岛惩罚（中心高但邻居低）
            from scipy.ndimage import convolve
            kernel = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]]) / 8.0
            neighbor_avg = convolve(smoothed_map, kernel, mode='constant')
            is_peak = smoothed_map > (max_activity * 0.8)
            is_isolated = (smoothed_map > neighbor_avg * 2) & is_peak
            island_count = np.sum(is_isolated)
            island_penalty = 1.0 / (1.0 + island_count * 0.5)  # 温和惩罚

            # 5. 综合得分
            concentration = peak_ratio * spatial_focus * bin_penalty * island_penalty
            firing_map_norm = smoothed_map / np.sum(smoothed_map)

            # 计算与位置先验的相似性
            similarity_to_prior = 1 - cosine(firing_map_norm.flatten(), position_prior_map.flatten())
            # 惩罚因子（可调参数）
            prior_penalty = max(0.1, 1.0 - min(1.0, max(0, similarity_to_prior - 0.6) * 2.5))

            # 最终得分
            score = max(0.01, stability - 0.8)*5 * (concentration ** 0.5) * prior_penalty ** 2
            return {
                'offset': offset,
                # 'map': clustedActivityData,
                'stability': stability,
                'concentration': concentration,
                'similarity_to_prior': similarity_to_prior,
                'prior_penalty': prior_penalty,
                'score': score
            }

        except Exception as e:
            print(f"❌ Offset {offset} processing failed: {str(e)}")
            return None  # 或者返回默认值

if __name__ == '__main__':
    root=Tk()
    root.withdraw()
    CreatedDataPerDayInstances:dict = {}

    while True:
        _cur, _instance = CreateDayInstance()
        CreatedDataPerDayInstances[_cur] = _instance
        if input("是否继续？(y/n)") == "n":
            break