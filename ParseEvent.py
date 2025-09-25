
import numpy as np

def read_event_file(event_fn):
    import os
    import struct
    # 获取文件信息和文件大小
    file_info = os.stat(event_fn)
    file_size = file_info.st_size

    EVENTSZ = 12
    event_count = (file_size // EVENTSZ) - 2

    if event_count > 0:
        vt_timestamps = [0] * event_count
        ttls = [0] * event_count
        with open(event_fn, 'rb') as fid:
            fid.seek(EVENTSZ, os.SEEK_SET)
            for n_event in range(event_count):
                vt_timestamps[n_event], ttls[n_event] = struct.unpack('QI', fid.read(EVENTSZ))
        st_event_data = {
            'vtTimestamps': vt_timestamps,
            'TTLs': ttls
        }
        return st_event_data
    else:
        return None



def extract_event(st_event_data, bits):
    import numpy as np
    bit_val = [bit >> bits[0] & 1 for bit in st_event_data['TTLs']]
    init_val = bits[1]
    de_bit_val = np.diff(np.concatenate(([init_val], bit_val)))

    if init_val == 0:
        lg_s = [i for i, x in enumerate(de_bit_val) if x == 1]
        lg_e = [i for i, x in enumerate(de_bit_val) if x == -1]
    else:
        lg_s = [i for i, x in enumerate(de_bit_val) if x == -1]
        lg_e = [i for i, x in enumerate(de_bit_val) if x == 1]
    event = {
        'Time_S': [st_event_data['vtTimestamps'][i] for i in lg_s],
        'Time_E': [st_event_data['vtTimestamps'][i] for i in lg_e]
    }
    return event



def ParseEvent(event_fn, st_events_bits=None):
    if st_events_bits is None:
        #0：低到高为S, 高到低为E
        st_events_bits = {
            'OgEnable'  :[2 , 0],
            'MsEnable' :[3 , 0],
            'MsSync' :[4, 0],
            'Lick0'  :[5, 0],
            'Pump0'  :[6, 0],
            'lick1'  :[7, 0],
            'VisualStimulate'  :[8, 0],
            'BaslerSync'  :[9, 0],
            # 'ModR'  :[12, 0],
            # 'MinS'  :[9 , 1],
            # 'MinT'  :[15, 0]
        }

    st_event_data = read_event_file(event_fn)
    st_events = {}

    for event_field in st_events_bits:
        st_events[event_field] = extract_event(st_event_data, st_events_bits[event_field])
    return st_events
    
if __name__ == '__main__':
    import os
    event_fn = r'C:\Users\KEHAIXING\Downloads\lyf_01.evt'
    if os.path.exists(event_fn):
        evt = ParseEvent(event_fn)
        for key in evt:
            print(key, len(evt[key]['Time_S']))
        print(evt)
        

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