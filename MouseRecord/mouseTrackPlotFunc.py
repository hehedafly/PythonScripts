import os 
import sys
import json
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import colorsys
import math
import time
import shutil
import bisect
import pickle
import multiprocessing
import threading
from collections import defaultdict
import matplotlib.patches as patches
from io import StringIO
from typing import List, Tuple, Dict
from multiprocessing import Pool, Process, Queue
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import hsv_to_rgb
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation

from math import*
from ParseEvent import ParseEvent
ignoreFristTrials = 15
ignoreLastTrials = 15
maxTrials = ignoreFristTrials + 320 + ignoreLastTrials

global trialGroupCount, trialPerGroup
trialGroupCount= -1
trialPerGroup = 2 if trialGroupCount < 0 else -1

def Distance(pos1:list, pos2:list) -> float:
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def Direction(pos1:list, pos2:list) -> float:
    return degrees(atan2(pos2[1] - pos1[1], pos2[0] - pos1[0]))

def GetFileCreateTime(filePath:str) -> str:
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File {filePath} not found")
    
    return time.strftime("%m_%d", time.localtime(os.path.getmtime(filePath)))

def getFilesInFolder(folderPath:str, suffix:str = ""):
    fileList = []
    for root, _, files in os.walk(folderPath):
        for file in files:
            filePath = os.path.join(root, file)
            if (file.lower().endswith(suffix) or suffix == "") and os.path.isfile(filePath):
                fileList.append(filePath)
    return fileList

def GeneratePklFileName(_type:str, _mouseNmae:str = '', _day:int = -1, date:str = '', headerOnly = False, _suffix = None):
    defaultHeader = {"rec": "rec_raw_res_", "pos": "pos_raw_res_", "evt": "evt_raw_res_", "ignore": "", "summary": ""}
    defaultSuffix = {"rec": ".pkl", "pos": ".pkl", "evt": ".evt", "ignore": "", "summary": "_summary.pkl"}
    if _type not in ["rec", "pos", "evt", "ignore", "summary"]:
        raise ValueError("Invalid type")
    if not headerOnly and (_mouseNmae == '' or _day == -1):
        raise ValueError("Mouse name and day must be provided")
    
    header = f"{date + ("_" if date != '' else '')}"+defaultHeader[_type]
    suffix = _suffix if (_suffix != None and _suffix != '') else defaultSuffix[_type]
    if headerOnly:
        return header
    else:
        return header + f"{_mouseNmae}_day{_day}{suffix}"

def GetPklFile(fileList:list[str], _type:str, _mouseName:str, _day:int) -> str:
    name = GeneratePklFileName(_type, _mouseName, _day, date='')

    relatedFiles = [x for x in fileList if name in x ]
    if len(relatedFiles) == 0:
        print(f"Warning: no file found for {_type} of {_mouseName} on day {_day}")
        return ""
    if len(relatedFiles) == 1:
        return relatedFiles[0]
    else:
        print(f"Warning: {len(relatedFiles)} files found for {_type} of {_mouseName} on day {_day}, frist is taken:\n {relatedFiles}")
        return relatedFiles[0]
    
def GetFormatedFileInfo(pklFileName, extraSuffix:str = ''):
    suffix = "." + pklFileName.split(".")[-1]
    date = ''
    if '_raw_res_' in pklFileName:
        _type:str = pklFileName[:pklFileName.index("_raw_res_")]
    else:
        _type = "ignore"
        date = ""

    if "_" in _type:
        date = "_".join(_type.split("_")[:-1])
        _type = _type[len(date) + 1:]  
        name = pklFileName[pklFileName.index("raw_res_") + len("raw_res_"): pklFileName.index("_day")]
    else:
        name = pklFileName[: pklFileName.index("_day")]

    day = int(pklFileName[pklFileName.index("_day") + len("_day"): pklFileName.index(suffix) - len(extraSuffix)])
    return date, _type, name, day

def auto_align_with_ratio(ref, noisy, ratio, tolerance):
    if len(ref) == 0 or len(noisy) == 0:
        print("Empty input")
    
    ref = np.asarray(ref)
    noisy = np.asarray(noisy)
    refdiff:np.ndarray = np.diff(ref) * ratio
    # noisydiff = np.diff(noisy)
    refdiffReverse = np.flip(refdiff)

    diffDiff = []
    for noise in noisy[(len(ref) - 1):]:
        tempRes = [noise]
        minused = noise
        for rdiff in refdiffReverse:
            minused = minused - rdiff
            tempRes.append(minused)

        diffDiff.append(np.flip(tempRes))

    indicesMatch = []
    indicesMatchDiff = []
    for noisyMinusDiff in diffDiff:
        tempindicesMatch = []
         
        for minusedDiff in noisyMinusDiff:
        # 检查数据数组中是否存在一个元素在 ref ± tolerance 范围内
            found = any(abs(d - minusedDiff) <= tolerance for d in noisy)
            if not found:
                break
        else:
            for rightMinusedDiff in noisyMinusDiff:
                indices = np.asarray([i for i, d in enumerate(noisy) if abs(d - rightMinusedDiff) <= tolerance])
                if len(indices) == 1:
                    tempindicesMatch.append(indices[0])
                else:
                    matchdiff = abs(noisy[indices] - rightMinusedDiff)#对当前潜在预测找到原始数据中最符合潜在预测的值
                    tempindicesMatch.append(indices[np.where(matchdiff == min(matchdiff))[0][0]])
        indicesMatch.append(tempindicesMatch)
        if len(tempindicesMatch) == len(ref):
            indicesMatchDiff.append(np.sum(abs(noisy[np.array(tempindicesMatch)] - ref * ratio)))
        else:
            indicesMatchDiff.append(-1)
    
    indicesMatchDiff = np.asarray(indicesMatchDiff)
    if np.all(indicesMatchDiff < 0):
        print("No match found")
    bestMatchInd = np.where(indicesMatchDiff == min(indicesMatchDiff[indicesMatchDiff >= 0]))[0][0]
    bestMatch = indicesMatch[bestMatchInd]
    
    return bestMatch


def AlignProjectBackToUnityTime(timesInLogevent, offsetInLogevent, scale = 50000)-> np.ndarray:
    return (np.asarray(timesInLogevent) - offsetInLogevent) / scale

def ResampleByStartTime(eventStart, eventIndex, manipulateStart, manipulateEnd, tolerance = 0.2):
    """
    找出控制时间段内发生的目标事件及其标签
    
    参数:
        eventStart: 目标事件开始时间列表
        eventIndex: 目标事件标签列表(与target_starts长度相同)
        manipulateStart: 控制事件开始时间列表
        manipulateEnd: 控制事件结束时间列表
    
    返回:
        一个列表，每个元素是一个元组，包含:
        (控制时间段索引, 该控制时间段内发生的目标事件索引列表, 对应的标签列表)
    """
    # 首先确保输入数据长度匹配
    assert len(eventStart) == len(eventIndex), "目标事件开始时间和标签列表长度不一致"
    assert len(manipulateStart) == len(manipulateEnd), "控制事件开始和结束时间列表长度不一致"
    
    result = []
    
    # 遍历每个控制时间段

    for control_idx, (ctrl_start, ctrl_end) in enumerate(zip(manipulateStart, manipulateEnd)):
        # 检查时间有效性
        if ctrl_start > ctrl_end:
            continue  # 或者可以抛出异常，取决于你的需求
            
        # 存储当前控制时间段内的目标事件索引和标签
        target_indices = []
        # half_target_indices = []
        target_labels_in_period = []
        
        # 检查每个目标事件是否在当前控制时间段内
        for target_idx, (t_start, label) in enumerate(zip(eventStart, eventIndex)):
            if t_start - ctrl_start > tolerance * -1 and t_start < ctrl_end:
                target_indices.append(target_idx)
                target_labels_in_period.append(label)
            # elif target_idx :
        # 添加到结果中
        # result.append((control_idx, target_indices, target_labels_in_period))
        result += target_labels_in_period
    
    return result

def auto_segment(data_points: np.ndarray, 
                               min_interval: float,
                               outlier_threshold: float = 2.0) -> List[np.ndarray]:
    """
    基于最小间隔检测离群点并进行分段
    
    参数:
    time_points: 一维时间点数组
    min_interval: 最小允许的时间间隔
    outlier_threshold: 离群点判断阈值（相对于平均间隔的倍数）
    返回:
    分段的时间点列表
    """
    time_points = data_points

    assert len(time_points) > 0, "时间点数组不能为空"
    assert min_interval > 0, "最小间隔必须大于0"
    
    # 将时间点排序并计算间隔
    sorted_times = np.sort(time_points)
    intervals = np.diff(sorted_times)
    
    if len(intervals) == 0:
        return []
    
    # 计算间隔统计信息
    # mean_interval = np.mean(intervals)
    median_interval = np.median(intervals)
    # std_interval = np.std(intervals)
    
    # 检测离群点（异常大的间隔）
    outlier_mask = intervals > max(min_interval, median_interval * outlier_threshold)
    outlier_indices = np.where(outlier_mask)[0]
    
    # 获取离群点本身（间隔后的点）
    outlier_points = []
    for idx in outlier_indices:
        outlier_points.append(sorted_times[idx + 1])  # 间隔后的点很可能是离群点
    
    # 根据异常间隔进行分段
    segments = []
    current_segment = [sorted_times[0]]
    
    for i in range(len(intervals)):
        if outlier_mask[i]:
            # 发现异常间隔，结束当前段
            segments.append(np.array(current_segment))
            current_segment = [sorted_times[i + 1]]
        else:
            current_segment.append(np.array(sorted_times[i + 1]))
    
    if current_segment:
        segments.append(np.array(current_segment))
    
    return segments

def adaptive_segmentation(time_points: List[float], 
                         min_interval: float,
                         density_threshold: float = 0.5) -> Dict:
    """
    自适应分段方法，考虑点密度
    
    参数:
    time_points: 一维时间点数组
    min_interval: 最小允许的时间间隔
    density_threshold: 密度阈值（点数/时间范围）
    
    返回:
    分段结果和统计信息
    """
    sorted_times = np.sort(time_points)
    
    segments = []
    current_segment = [sorted_times[0]]
    outliers = []
    
    for i in range(1, len(sorted_times)):
        current_interval = sorted_times[i] - sorted_times[i-1]
        
        if current_interval > min_interval:
            # 检查当前段的密度
            segment_duration = current_segment[-1] - current_segment[0] if len(current_segment) > 1 else 0
            segment_density = len(current_segment) / max(segment_duration, 1e-10)
            
            if segment_density < density_threshold and len(current_segment) > 1:
                # 低密度段，可能包含离群点
                outliers.extend(current_segment[1:])  # 保留第一个点作为正常点
                current_segment = [current_segment[0], sorted_times[i]]
            else:
                # 正常分段
                segments.append(current_segment)
                current_segment = [sorted_times[i]]
        else:
            current_segment.append(sorted_times[i])
    
    if current_segment:
        segments.append(current_segment)
    
    return {
        "segments": segments,
        "outliers": outliers,
        "outlier_count": len(outliers),
        "segment_count": len(segments)
    }


def segment_time_points(time_points: np.ndarray, num_segments: int) -> list[np.ndarray]:

    if not time_points or num_segments <= 0:
        return []
    
    if num_segments == 1:
        return [time_points]
    
    sorted_times = np.sort(time_points)
    
    # 使用等间距的初始中心点
    total_duration = sorted_times[-1] - sorted_times[0]
    centers = [sorted_times[0] + (i + 0.5) * total_duration / num_segments 
              for i in range(num_segments)]
    
    # 分配每个时间点到最近的中心点
    segments = [[] for _ in range(num_segments)]
    
    for time_point in sorted_times:
        # 找到最近的中心点
        distances = [abs(time_point - center) for center in centers]
        nearest_idx = np.argmin(distances)
        segments[nearest_idx].append(time_point)
    
    return segments

def ResampleByGivenStartAndEnd(trialstartime,trialfinishtime,OGstartime,OGfinishtime,threshold,Ratio,mode):
    assert len(trialstartime) == len(trialfinishtime)
    assert len(OGstartime) == len(OGfinishtime)
    assert mode in [0,1,2,3]
    
    temp_timearray = np.minimum(np.array(trialfinishtime)[: ,np.newaxis] ,np.array(OGfinishtime)[np.newaxis, :]) - np.maximum(np.array(trialstartime)[: ,np.newaxis] ,np.array(OGstartime)[np.newaxis, :]) 
    temp_timearray[temp_timearray < 0] = 0
    SumTime = np.sum(temp_timearray,axis=1)
    SumTimeMask = SumTime > threshold
    RatioMask = SumTime/(np.array(trialfinishtime) - np.array(trialstartime)) > Ratio
    Mask = None
    if mode == 0:     
        Mask = SumTimeMask
    elif mode == 1:
        Mask = RatioMask
    elif mode == 2:
        Mask = SumTimeMask & RatioMask
    elif mode == 3:
        Mask = SumTimeMask | RatioMask
    return np.array(range(len(trialstartime)))[Mask]

def generate_rainbow_colors(num_colors):
            """
            生成指定长度的彩虹色列表，格式为'#RRGGBB'
            
            参数:
                num_colors (int): 需要生成的颜色数量
            
            返回:
                list: 包含彩虹色hex字符串的列表
            """
            colors = []
            for i in range(num_colors):
                # 在HSV色彩空间中均匀分布色相(0-1)，保持饱和度和亮度为1
                hue = 0.8 - (i / num_colors) * 0.8
                # 将HSV转换为RGB (所有值在0-1范围内)
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                # 将RGB值从0-1转换为0-255并转换为16进制
                hex_color = "#{:02X}{:02X}{:02X}".format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors.append(hex_color)
            return colors

def compute_curvature(points):
    """计算轨迹的平均曲率"""
    if len(points) < 3:
        return 0.0, 0.0
    
    curvatures = []
    for i in range(1, len(points)-1):
        p0 = points[i-1]
        p1 = points[i]
        p2 = points[i+1]
        
        # 计算三点曲率
        vec1 = p1 - p0
        vec2 = p2 - p1
        cross = np.cross(vec1, vec2)
        denominator = np.linalg.norm(vec1)**3 + np.linalg.norm(vec2)**3
        curvatures.append(abs(cross)/(denominator + 1e-6))
    
    return np.mean(curvatures), np.std(curvatures)

def extract_features(dataDict:dict, indice, destAngle, sceneInfo):
    """核心特征提取函数"""
    features = {}
    features['fristhalflength'] = dataDict['fhDistances'][indice]
    
    # segDir = []
    # segLength = 0
    # fristValidDir = -1
    # trajectorySeg = []
    threshold_distance = sceneInfo[2] * 0.25
    target_distance = sceneInfo[2] - 115
    fhdir = -1
    lhdir = -1
    distanceBetween = 0

    # if indexPath is not None:
    fullTraj = dataDict['fhTrackFlat'][indice]
    fullTraj = fullTraj.loc[:, 'x':'y'].values

    fullpoints = np.asarray(fullTraj, dtype=float).T
    center = sceneInfo[0:2]
    
    # 计算每个点到起始点和结束点的距离
    dist_from_start = np.linalg.norm(fullpoints - center.reshape(2, 1), axis=0)
    dist_from_end = np.linalg.norm(fullpoints - center.reshape(2, 1), axis=0)

    # plt.plot(fullpoints[0], fullpoints[1])
    # plt.annotate(f"{max(dist_from_start)}", (np.mean(fullpoints[0]), np.mean(fullpoints[1])))
    
    # 找到第一段的起始索引（超过阈值）
    start_mask = dist_from_start >= threshold_distance
    start_indices = np.where(start_mask)[0]
    if not start_indices.size:
        print(f"{indice} angle:{destAngle} 没有轨迹点超过起始点的阈值距离。")
        # plt.show()
        return features
    first_start_idx = start_indices[0]
    
    # 找到第一段的结束索引（达到目标距离）
    after_start_dists = dist_from_start[first_start_idx:]
    target_mask = after_start_dists >= target_distance
    target_indices = np.where(target_mask)[0]
    if not target_indices.size:
        print(f"{indice} angle:{destAngle} 起始点附近没有达到目标距离的轨迹点。")
        # plt.show()
        return features

    first_end_idx = first_start_idx + target_indices[0]
    
    # 找到最后段的起始索引（距离结束点超过目标距离的最后一个点）
    end_mask = dist_from_end >= target_distance
    end_indices = np.where(end_mask)[0]
    if not end_indices.size:
        print(f"{indice} angle:{destAngle} 结束点附近没有超过目标距离的轨迹点。")
        # plt.show()
        return features
    last_start_idx = end_indices[-1]
    
    # 检查索引有效性
    if last_start_idx <= first_end_idx:
        print(f"{indice} angle:{destAngle} 最后段的起始点在第一段结束点之前，无法分割。")
        # plt.show()
        return features
    
    # 分割轨迹
    segment1 = fullpoints[:, first_start_idx : first_end_idx + 1]
    # segment2 = fullpoints[:, first_end_idx + 1 : last_start_idx]
    segment3 = fullpoints[:, last_start_idx :]

    dir1 = degrees(np.arctan2(segment1[1, -1] - segment1[1, 0], segment1[0, -1] - segment1[0, 0]))
    dir2 = degrees(np.arctan2(segment3[1, -1] - segment3[1, 0], segment3[0, -1] - segment3[0, 0]))
    distanceBetween = np.sum(np.sqrt((np.gradient(fullpoints[0])) **2 + (np.gradient(fullpoints[1]))**2)[first_end_idx+1 : last_start_idx])
    fhdir = 270 + dir1 - destAngle
    fhdir = (fhdir + 360) % 360
    fhdir = fhdir if fhdir < 180 else (360 - fhdir) * -1
    lhdir = 270 + dir2 - destAngle
    lhdir = (lhdir + 360) % 360
    lhdir = lhdir if lhdir < 180 else (360 - lhdir) * -1

    # plt.scatter(segment1[0, 0], segment1[1, 0], c='blue')
    # plt.scatter(segment1[0, -1], segment1[1, -1], c='green')
    # plt.scatter(segment3[0, 0], segment3[1, 0], c='y')
    # plt.scatter(segment3[0, -1], segment3[1, -1], c='black')
    # plt.show()
        # trajectorySeg.append(segment1)

    features['fristHalfDirDiff'] = fhdir
    features['lastHalfDirDiff'] = lhdir
    features['distanceBetween'] = distanceBetween

    return features

def ResampleTrack(points:np.ndarray, num = 100) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((num, 2))
    
    # 去除连续重复点
    if len(points) > 1:
        mask = np.any(np.diff(points, axis=0) != 0, axis=1)
        keep = np.concatenate([[True], mask])
        points = points[keep]
    
    n_points = len(points)
    if n_points == 1:
        return np.tile(points[0], (num, 1))
    
    # 计算各段距离及累积距离
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_dists[-1]
    
    if total_length == 0:
        return np.tile(points[0], (num, 1))
    
    # 生成目标弧长
    s_target = np.linspace(0, total_length, num)
    
    # 确定每个目标弧长对应的段索引
    indices = np.searchsorted(cum_dists, s_target, side='right') - 1
    indices = np.clip(indices, 0, len(dists) - 1)
    
    # 计算线段内插值比例
    s_in_segment = s_target - cum_dists[indices]
    segment_lengths = dists[indices]
    t = s_in_segment / segment_lengths  # 由于预处理过，segment_lengths不会为0
    
    # 线性插值计算坐标
    start_points = points[indices]
    end_points = points[indices + 1]
    interpolated = start_points + t[:, np.newaxis] * (end_points - start_points)
    
    return interpolated

class TrajectoryMatcher:

    """轨迹匹配器"""
    def __init__(self, n_neighbors=10, metric='manhattan'):
        self.scaler = RobustScaler()
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.feature_names = []
        
    def fit(self, indexPaths, dirLs):
        global features
        # 特征提取
        feature_list = []
        for i in range(len(indexPaths)):
            feat = extract_features(indexPaths[i], dirLs[i])
            tempFeatDict:dict = copy.deepcopy(feat)
            tempFeatDict['index'] = indexPaths[i]
            features.append(tempFeatDict)
            feature_list.append([feat.get(k, 0) for k in list(feat.keys())])
            if i % 2000 == 0:
                print(f"已处理 {i} 个轨迹")
        else:
            print(f"已处理 {len(indexPaths)} 个轨迹")

        self.feature_names = list(feat.keys())
        # 标准化
        self.X = np.array(feature_list)
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 构建索引
        self.nbrs.fit(self.X_scaled)
        return self
    
    def query(self, _index, _dir, neighbors=10):
        # 提取查询轨迹特征
        # feat = extract_features(query_traj, indices, query_mark, fullTraj)
        feat = extract_features(_index, _dir)

        query_vec = np.array([feat.get(k, 0) for k in self.feature_names]).reshape(1, -1)
        query_scaled = self.scaler.transform(query_vec)
        
        # 执行查询
        distances, indices = self.nbrs.kneighbors(query_scaled, neighbors)
        return indices[0], distances[0]

def flatten_nested_structure(nested_list, dtype, isList = False, oneDemArrSpread = False):
    res = []
    # dataframes = []
    # ndArrays = []
    indices = []
    
    def recursive_flatten(element, path, dtype):
        if isinstance(element, list) or (oneDemArrSpread and isinstance(element, np.ndarray) and element.ndim == 1):
            if isList and type(element[0]) == dtype:
                res.append(element)
                indices.append(path)
            else:
                for idx, item in enumerate(element):
                    recursive_flatten(item, path + (idx,), dtype)
        elif dtype == pd.DataFrame and isinstance(element, pd.DataFrame):
            res.append(np.asarray([element['x'].values, element['y'].values], dtype=int))
            indices.append(path)
        elif isinstance(element, dtype):
            res.append(element)
            indices.append(path)
        else:
            raise TypeError("Unsupported element type: {}".format(type(element)))
    
    recursive_flatten(nested_list, (), dtype)
    return res, indices

def rotate_points(points, center_x, center_y, angle_degrees, clockwise=False):
    """
    将轨迹点绕指定中心点旋转指定角度
    
    参数:
        points: 轨迹点列表，格式为[(x1, y1), (x2, y2), ...]
        center_x: 中心点的x坐标
        center_y: 中心点的y坐标
        angle_degrees: 旋转角度（度数）
        clockwise: 是否顺时针旋转，默认为False（逆时针）
    
    返回:
        旋转后的轨迹点列表
    """
    
    angle_rad = math.radians(angle_degrees)
    if clockwise:
        angle_rad = -angle_rad  # 顺时针转换为负角度
    
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    rotated_points = []
    for pos in points:
        x = pos[0]
        y = pos[1]
        # 平移到原点
        x_translated = x - center_x
        y_translated = y - center_y
        
        # 应用旋转矩阵
        new_x = x_translated * cos_theta - y_translated * sin_theta
        new_y = x_translated * sin_theta + y_translated * cos_theta
        
        # 平移回原坐标系
        new_x += center_x
        new_y += center_y
        
        rotated_points.append((new_x, new_y))
    
    return rotated_points

class Day:
    def __init__(self, _RecMainDfPath:str, _PosMainDfPath:str, _name:str, _day:int, _eventFile = '', _savePath = ''):
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
        self.bkgPosTrack:pd.DataFrame= None
        self.RecMainDf:pd.DataFrame = None
        self.PosMainDf:pd.DataFrame = None
        _RecMainDfPath = _RecMainDfPath.replace('\\', '/')
        _PosMainDfPath = _PosMainDfPath.replace('\\', '/')
        self.MainDfSavePath = [_RecMainDfPath, _PosMainDfPath]
        if os.path.exists(_RecMainDfPath) and os.path.exists(_PosMainDfPath):
            self.RecMainDf = pd.read_pickle(_RecMainDfPath)
            self.PosMainDf = pd.read_pickle(_PosMainDfPath)
            _sceneInfo = json.loads(self.PosMainDf['sceneInfo'].values[0])
            _selectedAreas = json.loads(self.PosMainDf['sceneInfo'].values[1])
            # if len(_selectedAreas) and len(_sceneInfo):
            #     sceneInfo.append(_sceneInfo)
            #     selectedAreas.append(_selectedAreas)
        # self.RecMainDfClear:pd.DataFrame = _RecMainDf
        # self.PosMainDfClear:pd.DataFrame = None
        self.trialPerGroup = -1
        self.day = _day
        self.name = _name
        self.eventFile = _eventFile
        self.fullEventRecord = None
        self.eventType = ['OGManipulate', 'miniscopeRecord']
        self.deviceEventRecord = {}
        self.trialElapsedTimeMean = None
        self.trialIntervalTimeMean = None
        self.rescale = -1
        self.offset = 0

        self.savePath = _savePath if _savePath!= '' else os.path.join(os.getcwd(), 'tempSummaryLogs')
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def Process(self): 
        if self.RecMainDf is None or self.PosMainDf is None:
            raise ValueError("RecMainDf or PosMainDf is None")
        
        print("Processing mouse " + self.name + " day " + str(self.day))
        init = self.RecMainDf.index[self.RecMainDf['type'] == "init"]
        if (len(init)):
            init = max(init)
        else:
            init = 0
        self.RecMainDf = self.RecMainDf[init:]

        self.startIndices    =          self.RecMainDf.index[self.RecMainDf['type'] == "start"]#trial处理均以开始结束Index为标准
        self.EndIndices      =          self.RecMainDf.index[self.RecMainDf['type'] == "end"]
        # self.licks           = np.array(self.RecMainDf.loc[self.RecMainDf["type"] == "lick"]["delta time"])
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
        time = self.PosMainDf.loc[self.PosMainDf['unitDistance'] > 100,'TimeInUnitySecFromTrialStart']
        print(f'warning:basler卡顿时间点数量:{len(time)}')

        _startTime = self.RecMainDf.loc[self.startIndices[0]]['delta time']
        self.bkgPosTrack = self.PosMainDf.loc[self.PosMainDf['TimeInUnitySecFromTrialStart'] < _startTime]
        segment = []
        fristIgnoredTrials = 0
        lastIgnoredTrials = 0

        for event in self.eventType:
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
                self.licks.append(             self.trials[-1].loc[self.trials[-1]['type'] == "lick"]['delta time'].values)
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

                tempStayTime = tempDf.loc[tempDf['type'] == 'stay']['delta time'].values
                tempStayTime = tempStayTime[0] if len(tempStayTime) else -1
                self.trialPosMark.append([self.GetPosOrIndAtTime(tempStayTime - 0.5), self.GetPosOrIndAtTime(tempStayTime - 0.25)])
                self.trialfhElaspedTime.append(tempStayTime - tempStartTime)
                self.fhDistancePerTrial.append(np.sum(unitDistance[self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempStayTime)]]))
                self.trialTimeMark.append([self.GetPosOrIndAtTime(tempStartTime, self.posPerTrial[-1], needInd=True), self.GetPosOrIndAtTime(tempStayTime - 0.5, self.posPerTrial[-1], needInd=True), self.GetPosOrIndAtTime(tempStayTime, self.posPerTrial[-1], needInd=True)])
                self.trialTimeMarkCollected.append([tempStartTime, tempStayTime, tempEndTime])
                # dir, polar = self.TrackNormalization(np.asarray([self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values]).T, 20, 20)
                # self.trackDirection.append(tempDf.loc[tempDf['type'] == 'start']['lickPos'])
                # self.posPolarPerTrial.append(polar)
                # if self.day >=3:
                #     DrawTrack(self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values, self.unitDistancePerTrial[-1], addInfo=f'{self.name}_{self.day}_{np.sum(self.unitDistancePerTrial[-1])/sceneInfo[2]:.2f}_trial{i}', show=False, save=True, lw=0.6, pointSize=0.3)
                # DrawTrack(self.posPerTrial[-1]['smooth_x'].values, self.posPerTrial[-1]['smooth_y'].values, addInfo=f'Level{np.sum(self.unitDistancePerTrial[-1])//2500}_{self.name}_{self.day}_{i - 1}', show=False, save=True)
                
            else:
                posDfDropIndexes = self.PosMainDf.index[(self.PosMainDf['TimeInUnitySecFromTrialStart'] >= tempStartTime) & (self.PosMainDf['TimeInUnitySecFromTrialStart'] <= tempEndTime)]
                segment.append(posDfDropIndexes[0])
                self.trials.pop()
                # self.PosMainDfClear = self.PosMainDfClear.drop(posDfDropIndexes)
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
        if _endInd - _startInd < 15:
            _startInd = 0
            _endInd = fullTrialCount
        self.trials             = self.trials[              _startInd: _endInd]
        self.trialIndexes       = self.trialIndexes[        _startInd: _endInd]
        self.licks              = self.licks[               _startInd: _endInd]
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

        # self.licks = np.array(self.licks).flatten()
        # self.trackDirection        = self.trackDirection[         _startInd: _endInd]
        # self.posPolarPerTrial       = self.posPolarPerTrial[        _startInd: _endInd]
        if os.path.isfile(self.eventFile):
            self.fullEventRecord = ParseEvent(self.eventFile)

        SummaryDf = pd.DataFrame()
        SummaryDf['trials'              ] = self.trials
        SummaryDf['trialIndexes'        ] = self.trialIndexes
        SummaryDf['licks'               ] = self.licks
        SummaryDf['lickPos'             ] = self.licksPos
        SummaryDf['result'              ] = self.trialResults
        SummaryDf['trialElapsedTime'    ] = self.trialElapsedTime
        SummaryDf['trialfhElaspedTime'  ] = self.trialfhElaspedTime
        SummaryDf['trialIntervalTime'   ] = self.trialIntervalTime
        SummaryDf['posPerTrial'         ] = self.posPerTrial
        SummaryDf['uposPerTrial'        ] = self.uposPerTrial
        SummaryDf['speedPerTrial'       ] = self.speedPerTrial
        SummaryDf['fuDistancePerTrial'  ] = self.fuDistancePerTrial
        SummaryDf['fhDistancePerTrial'  ] = self.fhDistancePerTrial
        SummaryDf['uDistancePerTrial'   ] = self.uDistancePerTrial
        SummaryDf['DistancePerTrial'    ] = self.DistancePerTrial
        SummaryDf['trialPosMark'        ] = [json.dumps(list(mark)) for mark in self.trialPosMark]
        SummaryDf['trialTimeMark'       ] = [json.dumps(list(mark)) for mark in self.trialTimeMark]
        SummaryDf['trialTimeMarkCollected']=[json.dumps((mark)) for mark in self.trialTimeMarkCollected]
        eventTexts = {}
        for event in self.eventType:
            eventTexts[event] = self.deviceEventRecord[event].to_json(orient = 'records')
            if eventTexts[event] == None:
                eventTexts[event] = ''
        
        SummaryDf['addons'              ] = [self.eventFile, 
                                             json.dumps(self.MainDfSavePath),
                                             json.dumps(self.fullEventRecord), 
                                             json.dumps(eventTexts),
                                             self.PosMainDf['sceneInfo'].values[0],
                                             self.PosMainDf['sceneInfo'].values[1],
                                             self.bkgPosTrack.to_json(orient = 'records'),
                                             self.trialPerGroup,
                                             self.rescale,
                                             self.offset
                                             ] + [''] * (len(SummaryDf) - 10)
        fileSavePath = GeneratePklFileName('summary', self.name, self.day, _suffix = "_summary")
        SummaryDf[
            ['trialIndexes', 'lickPos', 'result', 'trialElapsedTime', 'trialfhElaspedTime', 'trialIntervalTime', 'fuDistancePerTrial', 'fhDistancePerTrial', 'uDistancePerTrial', 'DistancePerTrial', 'trialPosMark', 'trialTimeMark', 'trialTimeMarkCollected', 'addons']
            ].to_csv(f'{self.savePath}/{fileSavePath}.csv', index=True, header=True)

        SummaryDf.to_pickle(f'{self.savePath}/{fileSavePath}.pkl')
        backupPath = f'{self.savePath}/bak'
        if not os.path.exists(backupPath):
            os.makedirs(backupPath)
        _recName = os.path.join(backupPath, self.MainDfSavePath[0].split('/')[-1])
        if os.path.exists(_recName):
            shutil.move(_recName, _recName + '.bak')
        _posName = os.path.join(backupPath, self.MainDfSavePath[1].split('/')[-1])
        if os.path.exists(_posName):
            shutil.move(_posName, _posName + '.bak')
        shutil.move(self.MainDfSavePath[0], _recName)
        shutil.move(self.MainDfSavePath[1], _posName)
        self.RecMainDf.to_pickle(self.MainDfSavePath[0])
        self.PosMainDf.to_pickle(self.MainDfSavePath[1])

        self.SecondProcess()

    def SecondProcess(self):
        self.trialResults = np.int8(self.trialResults)
        self.licksPos = np.int16(self.licksPos)
        self.trialTimeMarkCollected = np.array(self.trialTimeMarkCollected)

        # self.lickInterval       = self.licks[1:] - self.licks[0: -1]
        # self.lickInterval       = self.lickInterval[self.lickInterval > 0]
        # self.lickIntervalLog    = np.log(1/self.lickInterval)

        if(self.trialPerGroup < 0):
            try:
                self.trialPerGroup = max(1, ceil(len(self.trials)/trialGroupCount))
            except:
                self.trialPerGroup = 10
        else:
            self.trialPerGroup = trialPerGroup

        for i in range(0, ceil(len(self.trials) / self.trialPerGroup)):
            self.trialAccuracyInGroup.append(   np.sum(self.trialResults[           i * self.trialPerGroup:  min(len(self.trialResults)             , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialResults)      - i * self.trialPerGroup))
            self.trialElapsedTimeInGroup.append(np.sum(self.trialElapsedTime[       i * self.trialPerGroup:  min(len(self.trialElapsedTime)         , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialElapsedTime)  - i * self.trialPerGroup))
            self.trialfhElaspedTimeInGroup.append(np.sum(self.trialfhElaspedTime[   i * self.trialPerGroup:  min(len(self.trialfhElaspedTime)       , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialfhElaspedTime) - i * self.trialPerGroup))
            self.trialIntervalInGroup.append(   np.sum(self.trialIntervalTime[      i * self.trialPerGroup:  min(len(self.trialIntervalTime)        , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.trialIntervalTime) - i * self.trialPerGroup))
            self.trialfhDistanceInGroup.append( np.sum(self.fhDistancePerTrial[     i * self.trialPerGroup:  min(len(self.fhDistancePerTrial)       , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.fhDistancePerTrial)- i * self.trialPerGroup))
            self.trialfuDistanceInGroup.append( np.sum(self.trialfuDistanceInGroup[ i * self.trialPerGroup:  min(len(self.trialfuDistanceInGroup)   , (i + 1) * self.trialPerGroup)]) / min(self.trialPerGroup, len(self.fuDistancePerTrial)- i * self.trialPerGroup))

        self.trialElapsedTimeMean    = np.mean(self.trialElapsedTime)
        self.trialIntervalTimeMean   = np.mean(self.trialIntervalTime)

        return True
    
    def Load(self) -> bool:
        if self.RecMainDf is not None and self.PosMainDf is not None:
            return False
        
        path = f'{self.savePath}/{GeneratePklFileName('summary', self.name, self.day)}'
        if os.path.exists(path):
            try:
                tempLoadedDf:pd.DataFrame = pd.read_pickle(path)
                self.trials                 = tempLoadedDf['trials'                 ].values.tolist()
                self.trialIndexes           = tempLoadedDf['trialIndexes'           ].values.tolist()
                self.licks                  = tempLoadedDf['licks'                  ].values.tolist()
                self.licksPos               = tempLoadedDf['lickPos'                ].values.tolist()
                self.trialResults           = tempLoadedDf['result'                 ].values.tolist()
                self.trialElapsedTime       = tempLoadedDf['trialElapsedTime'       ].values.tolist()
                self.trialfhElaspedTime     = tempLoadedDf['trialfhElaspedTime'     ].values.tolist()
                self.trialIntervalTime      = tempLoadedDf['trialIntervalTime'      ].values.tolist()
                self.posPerTrial            = tempLoadedDf['posPerTrial'            ].values.tolist()
                self.uposPerTrial           = tempLoadedDf['uposPerTrial'           ].values.tolist()
                # self.posSegPerTrial =     tempLoadedDf['posSegPerTrial'].value    s
                self.speedPerTrial          = tempLoadedDf['speedPerTrial'          ].values.tolist()
                self.fuDistancePerTrial     = tempLoadedDf['fuDistancePerTrial'     ].values.tolist()
                self.fhDistancePerTrial     = tempLoadedDf['fhDistancePerTrial'     ].values.tolist()
                self.uDistancePerTrial      = tempLoadedDf['uDistancePerTrial'      ].values.tolist()
                self.DistancePerTrial       = tempLoadedDf['DistancePerTrial'       ].values.tolist()
                self.trialPosMark           = tempLoadedDf['trialPosMark'           ].values.tolist()
                self.trialTimeMark          = tempLoadedDf['trialTimeMark'          ].values.tolist()
                self.trialTimeMarkCollected = tempLoadedDf['trialTimeMarkCollected' ].values.tolist()
                
                self.trialPosMark = [json.loads(mark) for mark in self.trialPosMark]
                self.trialTimeMark = [json.loads(mark) for mark in self.trialTimeMark]
                self.trialTimeMarkCollected = [json.loads(mark) for mark in self.trialTimeMarkCollected]
                self.eventFile              = tempLoadedDf['addons'].values[0]
                self.MainDfSavePath         = json.loads(tempLoadedDf['addons'].values[1])
                self.fullEventRecord        = json.loads(tempLoadedDf['addons'].values[2])
                deviceEventRecordDict       = json.loads(tempLoadedDf['addons'].values[3])
                _sceneInfo                  = json.loads(tempLoadedDf['addons'].values[4])
                _selectedAreas              = json.loads(tempLoadedDf['addons'].values[5])
                self.bkgPosTrack            = pd.read_json(StringIO(tempLoadedDf['addons'].values[6]), orient='records')
                self.trialPerGroup          = int(tempLoadedDf['addons'].values[7]) if tempLoadedDf['addons'].values[7] != '' else -1
                self.rescale                = float(tempLoadedDf['addons'].values[8]) if tempLoadedDf['addons'].values[8] != '' else -1
                self.offset                 = float(tempLoadedDf['addons'].values[9]) if tempLoadedDf['addons'].values[9] != '' else 0
                # sceneInfo.append(_sceneInfo)
                # selectedAreas.append(_selectedAreas)
                for event in self.eventType:
                    if event in deviceEventRecordDict:
                        try:
                            self.deviceEventRecord[event] = pd.read_json(StringIO(deviceEventRecordDict[event]), orient='records')
                        except:
                            self.deviceEventRecord[event] = pd.DataFrame()
                    else:
                        self.deviceEventRecord[event] = pd.DataFrame()

                self.RecMainDf = pd.read_pickle(self.MainDfSavePath[0])
                self.PosMainDf = pd.read_pickle(self.MainDfSavePath[1])
                
                print(f"mouse {self.name} Day {self.day} loaded from file:{path}.")
                
                self.SecondProcess()
            except Exception as e:
                print(f"Error loading summary file {path}: {e}")
                return False
            return True
        else:
            return False
        
    def Clear(self):
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

    def CheckTrialIntegrity(self, tempDf:pd.DataFrame) -> bool:
        essentialTypes = ['stay']
        for essentialType in essentialTypes:
            if essentialType not in tempDf['type'].values:
                return False
        return True

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
            return int(results[0]) if singleResult else [int(r) for r in results]
        else:
            return df.loc[results].loc[:, 'x':'y'].values[0].tolist() if singleResult else df.loc[results]['x':'y'].values.tolist()
    
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

        # distancesToCenter = np.sqrt((x - sceneInfo[0])**2 + (y  - sceneInfo[1])**2)
        # # 创建掩码，排除首行
        # mask = distancesToCenter < sceneInfo[2]
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
    # 将时间序列分割为trials，顺序为上一个trial的结束到这个trial的结束
        weights = [0.1, 0.5, 0.4]
        
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
        
    # def DrawTrackPublic(self, _ax = None):
    #     DrawTrack(self.PosMainDf['smooth_x'].values, self.PosMainDf['smooth_y'].values, self.PosMainDf['speed'], title = "smoothed track", _ax = _ax)

    def ReturnLicks(self):
        return self.licks.copy()
    def ReturnTotalAccuracy(self):
        return self.trialResults.count(1)/len(self.trialResults)
    def ReturnResults(self):
        return self.trialResults.copy()
    def ReturnElaspedTime(self):
        return self.trialElapsedTime.copy()
    def ReturnTrialInterval(self):
        return self.trialIntervalTime.copy()
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
    def ReturnBackgroundPos(self):
        return self.bkgPosTrack.copy()
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
    def ReturnTrialTimeMarkCollected(self):
        return self.trialTimeMarkCollected.copy()
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
                self.dayInstance.Clear()
                self.dayInstance.Process()
    
    def __init__(self, _mouseName, _RecMainDfPklPaths, _PosMainDfPklPaths, _config, _trialPerGroup, _eventFiles = [], _multiThread = True, _savePath = ''):
        self.Days:list[Day] = []

        self.mouseName = _mouseName
        self.RecMainDfPklPaths = _RecMainDfPklPaths
        self.PosMainDfPklPaths = _PosMainDfPklPaths
        self.eventLogs = _eventFiles
        self.configs = _config
        self.trialPerGroup = _trialPerGroup
        self.multiThread = _multiThread
        self.savePath = _savePath if len(_savePath) else os.path.join(os.getcwd(), 'tempSummaryLogs')

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        
        threads = []
        for day in range(0, len(self.RecMainDfPklPaths)):
            self.Days.append(Day(self.RecMainDfPklPaths[day], self.PosMainDfPklPaths[day], self.mouseName, day, self.eventLogs[day], self.savePath))
        for day in self.Days:
            if self.multiThread:
                tempThread = self.DayProcess(day)
                tempThread.start()
                threads.append(tempThread)
            else:
                if not day.Load():
                    day.Clear()
                    day.Process()  
            # tempThread.join()
        for thread in threads:
            thread.join()
    
    def ReturnLicks(self):
        _tempLicksEveryDay = []
        for day in self.Days:
            _tempLicksEveryDay.append(day.ReturnLicks())
        return _tempLicksEveryDay

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
    
    def ReturnTrialInterval(self):
        _tempIntervalTimeEveryDay = []
        for day in self.Days:
            _tempIntervalTimeEveryDay.append(day.ReturnTrialInterval())
        return _tempIntervalTimeEveryDay
        
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
    
    def ReturnBackgroundPos(self):
        _tempBkgPosEveryDay = []
        for day in self.Days:
            _tempBkgPosEveryDay.append(day.ReturnBackgroundPos())
        return _tempBkgPosEveryDay

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
    
    def ReturnTrialTimeMarkCollected(self):
        _tempTimeMarkEveryDay = []
        for day in self.Days:
            _tempTimeMarkEveryDay.append(day.ReturnTrialTimeMarkCollected())
        return _tempTimeMarkEveryDay
    
    def ReturnEvents(self):
        _tempEventsEveryDay = []
        for day in self.Days:
            _tempEventsEveryDay.append(day.ReturnDeviceEvents())
        return _tempEventsEveryDay
    