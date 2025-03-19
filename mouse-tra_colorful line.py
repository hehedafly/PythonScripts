import os #检查文件是否存在
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from scipy.signal import savgol_filter

import tkinter as tk
from tkinter import filedialog
from math import*

def Distance(pos1:list, pos2:list) -> float:
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class Day:
    def __init__(self, _RecMainDf, _PosMainDf, _name, _day, _addition):
        self.trials                  :list= []
        self.startIndices            = []
        self.EndIndices              = []
        self.licks                   = []
        self.licksPos                :list= []
        self.lickInterval            :list= []
        self.lickIntervalLog         :list= []
        self.trialResults            :list= []
        self.trialElapsedTime        :list= []
        self.trialIntervalTime       :list= []
        self.trialAccuracyInGroup    :list= []
        self.trialElapsedTimeInGroup :list= []
        self.trialIntervalInGroup    :list= []
        
        self.posPerTrial             :list= []
        self.posSegPerTrial          :list= []

        self.RecMainDf:pd.DataFrame = _RecMainDf
        self.PosMainDf:pd.DataFrame = _PosMainDf
        self.trialPerGroup = -1
        self.day = _day
        self.name = _name

    def Process(self): 
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

        window_size = 7  # 根据实际情况调整
        self.PosMainDf['x'] = self.PosMainDf['x'].rolling(window_size, center=True, min_periods=1).median()
        self.PosMainDf['y'] = self.PosMainDf['y'].rolling(window_size, center=True, min_periods=1).median()

        # 2. 轨迹平滑（Savitzky-Golay滤波器）
        window_length = 15  # 必须为奇数，根据实际情况调整
        polyorder = 3       # 多项式阶数
        self.PosMainDf['x_smooth'] = savgol_filter(self.PosMainDf['x'], window_length, polyorder)
        self.PosMainDf['y_smooth'] = savgol_filter(self.PosMainDf['y'], window_length, polyorder)

        dt = np.gradient(self.PosMainDf['time'])  # 使用梯度处理非均匀时间戳
        dx = np.gradient(self.PosMainDf['x_smooth'])
        dy = np.gradient(self.PosMainDf['y_smooth'])
        self.speed = np.sqrt((dx/dt)**2 + (dy/dt)**2)
        self.PosMainDf['speed'] = self.speed
        
        for i in range(1, len(self.EndIndices)):
            self.trials.append(             self.RecMainDf.loc[self.startIndices[i-1] : self.startIndices[i], :])
            tempResult:int = int(self.trials[i - 1].loc[self.EndIndices[i - 1]]["result"])
            if tempResult >= 0:
                self.licksPos.append(           self.trials[i - 1].loc[self.EndIndices[i - 1] - 1]["lickPos"])
                self.trialResults.append(       self.trials[i - 1].loc[self.EndIndices[i - 1]]["result"])
                self.trialElapsedTime.append(   self.trials[i - 1].loc[self.EndIndices[i-1]]['delta time'] - self.trials[i - 1].loc[self.startIndices[i-1]]['delta time'])
                self.trialIntervalTime.append(  self.trials[i - 1].loc[self.startIndices[i]]['delta time'] - self.trials[i - 1].loc[self.EndIndices[i-1]]['delta time'])
                self.posPerTrial.append(self.PosMainDf.loc[tempStartTime <= self.PosMainDf['time'] <= tempEndTime, :])
                self.posSegPerTrial.append(self.SpeedSegment(self.posPerTrial['speed'], self.posPerTrial[i-1]))
            else:
                self.trials.pop()
            
            tempStartTime   = self.trials[i - 1].loc[self.startIndices[i-1]]['delta time']
            tempEndTime     = self.trials[i - 1].loc[self.EndIndices[i - 1]]['delta time']
        
            
        self.trials             = self.trials[              ignoreFristTrials : min(len( self.trials             - ignoreLastTrials), maxTrials)]
        self.licksPos           = self.licksPos[            ignoreFristTrials : min(len( self.licksPos           - ignoreLastTrials), maxTrials)]
        self.trialResults       = self.trialResults[        ignoreFristTrials : min(len( self.trialResults       - ignoreLastTrials), maxTrials)]
        self.trialElapsedTime   = self.trialElapsedTime[    ignoreFristTrials : min(len( self.trialElapsedTime   - ignoreLastTrials), maxTrials)]
        self.trialIntervalTime  = self.trialIntervalTime[   ignoreFristTrials : min(len( self.trialIntervalTime  - ignoreLastTrials), maxTrials)]
        self.posPerTrial        = self.posPerTrial[         ignoreFristTrials : min(len( self.trialIntervalTime  - ignoreLastTrials), maxTrials)]
        

        self.trialResults = np.int8(self.trialResults)
        self.licksPos = np.int8(self.licksPos)

        self.lickInterval       = self.licks[1:] - self.licks[0: -1]
        self.lickInterval       = self.lickInterval[self.lickInterval > 0]
        self.lickIntervalLog    = np.log(1/self.lickInterval)

        if(trialGroupCount > 0):
            self.trialPerGroup = ceil(len(self.trials)/trialGroupCount)
        else:
            self.trialPerGroup = trialPerGroup

        for i in range(0, ceil(len(self.trialResults) / self.trialPerGroup)):
            self.trialAccuracyInGroup.append(   np.sum(self.trialResults[       i * self.trialPerGroup:  min(len(self.trialResults)      , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialResults) - i * self.trialPerGroup))
            self.trialElapsedTimeInGroup.append(np.sum(self.trialElapsedTime[   i * self.trialPerGroup:  min(len(self.trialElapsedTime)  , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialResults) - i * self.trialPerGroup))
            self.trialIntervalInGroup.append(   np.sum(self.trialIntervalTime[  i * self.trialPerGroup:  min(len(self.trialIntervalTime) , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialResults) - i * self.trialPerGroup))

        self.trialElapsedTimeMean    = np.mean(self.trialElapsedTime)
        self.trialIntervalTimeMean   = np.mean(self.trialIntervalTime)

    def SpeedSegment(self, speedArray, posDf):
        speed_threshold = np.nanmean(speedArray) / 3
        low_speed_mask = speedArray < speed_threshold

        # 创建分段
        segments = []
        current_segment = []
        for i, is_low in enumerate(low_speed_mask):
            if not is_low:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(posDf.iloc[current_segment])
                    current_segment = []
        if current_segment:
            segments.append(posDf.iloc[current_segment])
        return segments

    def ReturnTotalAccuracy(self):
        return self.trialResults.count(1)/len(self.trialResults)
    def ReturnResults(self):
        return self.trialResults.copy()
    def ReturnElaspedTime(self):
        return self.trialElapsedTime.copy()
    def ReturnElaspedTimeInGroup(self):
        return self.trialElapsedTimeInGroup.copy()
    def ReturnAccuracyInGroup(self):
        return self.trialAccuracyInGroup.copy()
    def ReturnSpout(self):
        return self.licksPos.copy()
    def ReturnRightSpout(self):
        tempRes = self.trialResults * -1 + 1
        return np.abs(self.licksPos - tempRes)
    def ReturnMean(self):
        return {"trialElapsedTimeMean": self.trialElapsedTimeMean , "trialIntervalTimeMean": self.trialIntervalTimeMean}

class MouseDailyRecord:
    
    def __init__(self, _mouseInd, _RecMainDfs, _PosMainDfs, _config, _trialPerGroup, _addtion):
        self.Days:list[Day] = []

        self.mouseInd = _mouseInd
        self.RecMainDfs = _RecMainDfs
        self.PosMainDfs = _PosMainDfs
        self.configs = _config
        self.trialPerGroup = _trialPerGroup

        for day in range(0, len(self.RecMainDfs)):
            self.Days.append(Day(self.RecMainDfs[day], self.PosMainDfs[day], self.mouseInd, day, _addtion[day]))
        for day in self.Days:
            day.Process()    

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
        
root = tk.Tk()
root.withdraw()

ignoreFristTrials = 20
ignoreLastTrials = 20
maxTrials = ignoreFristTrials + 180

global trialGroupCount, trialPerGroup
trialGroupCount= 4
trialPerGroup = 100
mouseNames:list[str] = []
LogsFolders:list[str] = []
MouseRecords:list[MouseDailyRecord] = []
configs = []

All_folder_paths = filedialog.askdirectory(initialdir = os.getcwd(), title="选择包含指定批次Log数据的文件夹")
if not All_folder_paths:
    exit()

# 遍历每个选择的文件夹
for folder_path in os.listdir(All_folder_paths):
    if os.path.isdir(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(folder_path)
        LogsFolders.append(os.path.join(All_folder_paths, folder_path))
    elif os.path.isfile(os.path.join(All_folder_paths, folder_path)):
        mouseNames.append(All_folder_paths)
        LogsFolders.append(All_folder_paths)
        break


recfileList = []
posfileList = []
limitedColumns = ['type', 'delta time', 'mode', 'trial', 'lickPos', 'result']
for mouseIndex in LogsFolders:
    recfileList.append([])
    posfileList.append([])
    for file in os.listdir(mouseIndex):
        if file.endswith("rec.txt"):    
            recfileList[-1].append(file)
        elif file.endswith("pos.txt"):
            posfileList[-1].append(file)
    recfileList[-1].sort()
    posfileList[-1].sort()

def readToDataRows(_fname, _dataRows, _addtion:list = None):
    _config = ""
    with open(_fname, 'r') as file:
        for line in file:
            if line.strip() and "\t" in line:
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
            
AllMouseRecMainDfs = []
AllMousePosMainDfs = []
AllMouseAddLs = []
for mouseIndex in range(0, len(recfileList)):#fileList按mouse排序
    AllMouseRecMainDfs.append([])
    AllMousePosMainDfs.append([])
    AllMouseAddLs.append([])
    configs.append([])
    for DailyRecord in recfileList[mouseIndex]:
        _datarows = []
        _config = readToDataRows(LogsFolders[mouseIndex] + "/" + DailyRecord, _datarows, AllMouseAddLs[mouseIndex])
        configs[-1].append(_config)

        nowDf = pd.DataFrame(_datarows, columns=limitedColumns)
        nowDf = nowDf.drop(nowDf.index[nowDf['type'] == "type"])
        nowDf['delta time'] = np.float64(nowDf['delta time'])

        AllMouseRecMainDfs[-1].append(nowDf)

    for DailyPos in posfileList[mouseIndex]:
        _datarows = []
        nowDf = pd.read_csv(LogsFolders[mouseIndex] + "/" + DailyPos,sep='\t',names=['x','y', 'syncInd', "100*pythonTime", "frameInd", "TimeInUnitySecFromTrialStart"])
        sceneinfo = nowDf.iloc[1]
        nowDf = nowDf.drop([0, 1])
        nowDf = nowDf.drop_duplicates(subset='syncInd', keep='first')
        AllMousePosMainDfs[-1].append(nowDf)

for mouseIndex in range(0, len(mouseNames)):
    MouseRecords.append(
        MouseDailyRecord(mouseNames[mouseIndex], AllMouseRecMainDfs[mouseIndex], AllMousePosMainDfs[mouseIndex], configs[mouseIndex], trialPerGroup, AllMouseAddLs[mouseIndex])
    )



file_path = "D:/Lab/LAB/llaabb/Logs(new)/2025_01_19_19_24_pos.txt"
if not os.path.exists(file_path):
    print("文件不存在，请检查路径！")
else:
    print('文件存在，进行下一步')
    #从文件路径中提取文件名
    file_name = os.path.basename(file_path)
    #从文件名中提取日期部分
    date_part = file_name.split('_')[0:3]#提取前三个部分（年、月、日）
    date_str = '-'.join(date_part)#将年月日用'-'连接
    #读取数据文档
    data = pd.read_csv(file_path,sep='/t',usecols=[0,1],)#names=['x','y'],有表头时不用（-->如何优化？）
    
    
    #将数据提取出来
    x=data['x']
    y=data['y']
    #进行SG滤波：
    window_length = 7#窗口长度（必须为奇数）
    polyorder = 5#多项式阶数
    x_smooth = savgol_filter(x, window_length, polyorder)
    y_smooth = savgol_filter(y, window_length, polyorder)

#使用matplotlib.pyplot进行绘图
#创建绘图画布
    # plt.style.use('dark_background')#暗色背景
    plt.figure(figsize=(10,6))#画布大小
    #绘制原始运动轨迹线的代码
    # plt.plot(x, y, linestyle='-', linewidth=0.2, color='blue', alpha=0.4, label='OriginalTrajectory')
    # # 标记起点和终点
    # plt.scatter(x.iloc[0], y.iloc[0], s=5,facecolor='none',edgecolor='black', label='Start', zorder=3)
    # plt.scatter(x.iloc[-1], y.iloc[-1], s=5,facecolor='none', edgecolor='red', label='End', zorder=3)
    # 添加标题和标签
    plt.title(f'Mouse Movement Trajectory----{date_str}', fontsize=14)#使用提取的日期作为图的标题
    #背景中的网格，False为隐藏
    plt.grid(False)
    #调整坐标轴比例
    plt.axis('equal')#确保x和y轴比例相同，防止轨迹变形
    #绘制运动轨迹

  #不同方向的轨迹映射颜色：
    step=1
    for i in range(0,len(x_smooth)-step,step):#以step为步长，一直遍历数据到数据个数-步长
        dx = x_smooth[i+step]-x_smooth[i]
        dy = y_smooth[i+step]-y_smooth[i]#求两点之间的“距离”
        angle = np.arctan2(dy, dx)#计算方向角度
        angle = np.degrees(angle)%360
        #将角度映射到0-1范围（对应HSV的色相值）
        #手动生成高饱和度颜色
        hue = angle/360
        saturation=1.0#最大饱和度
        value=1.0#最大亮度
        rgb_color=hsv_to_rgb([hue,saturation,value])
        #绘制每一段滤波后的轨迹
        plt.plot(x_smooth[i:i + step + 1], y_smooth[i:i + step + 1], linestyle='-', linewidth=0.15, color=rgb_color, alpha=1)
    plt.scatter(x_smooth[0], y_smooth[0], s=5,facecolor='none',edgecolor='black', label='Start', zorder=3)
    plt.scatter(x_smooth[-1], y_smooth[-1], s=5,facecolor='none', edgecolor='red', label='End', zorder=3)
    plt.legend()
    plt.tight_layout()

    #生成带有日期和时间的文件名
    base_name = f'mouse_trajectory_{date_str.replace("-", "_")}'#文件名，如 "mouse_trajectory_2025_01_20"
    save_dir = os.path.dirname(file_path)#保存到原文件所在目录
    save_ext = '.png'#文件扩展名
    #生成数字后缀以区别
    counter=1
    while True:
        save_name=f'{base_name}_{counter}{save_ext}'#如 "mouse_trajectory_2025_01_20_1.png"
        save_path=os.path.join(save_dir,save_name)
        if not os.path.exists(save_path):#检查文件是否已存在
            break
        counter+=1#如果文件已存在,递增数字后缀
    #保存图片
    plt.savefig(save_path, dpi=900)#保存为图片
    print(f"图片已保存到 {save_path}")
    plt.show()