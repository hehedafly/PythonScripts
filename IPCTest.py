from multiprocessing import shared_memory

import cv2 as cv2
import numpy as np
from subprocess import call

class SharedMemoryObj:
    def __init__(self, name='shm_0', size=1024):
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)  # 尝试创建共享内存，若失败则映射同名内存空间
        except:
            self.shm = shared_memory.SharedMemory(name=name, create=False)

        self.shm_name = self.shm.name
        self.length = 0
        self.contentbegin = 0

    def write(self, _string:str) -> bool:
        try:
            if len(_string):
                self.shm.buf[0:len(_string)] = _string.encode()
                return True
            else:
                return False
        except Exception as e:
            traceback.print_exc()
            return False
    def writeBytes(self, _bytes:bytes) -> bool:
        try:
            if len(_bytes):
                self.shm.buf[0:len(_bytes)] = _bytes
                return True
            else:
                return False
        except Exception as e:
            traceback.print_exc()
            return False

    def read(self) -> str:
        try:
            length = int.from_bytes(bytes(self.shm.buf[:1]), "little")
            if length:
                tempResult = self.shm.buf[1:length+1]
                tempResult = tempResult.tobytes()
                tempResult = tempResult.decode()
                return tempResult
                
        except Exception as e:
            traceback.print_exc()
            return ""
        
if __name__ == '__main__':
    machine_config_shared_memory = SharedMemoryObj('UnityShareMemoryTest', 8 * 1024 * 1024)
    while True:
        machine_config_shared_memory.write("test")