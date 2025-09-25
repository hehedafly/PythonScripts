from ParseEvent import*
import numpy as np
from tkinter import Tk ,filedialog

root=Tk()
root.withdraw()

cur=list(filedialog.askopenfilenames(filetypes=[('logs', ('.evt'))]))
for evtFile in cur:
    logevent = ParseEvent(evtFile)
    print(f"Event file : {evtFile}")
    print(f"MsEnable : {len(logevent['MsEnable']['Time_S'])}")
    print(f"Pump0 : {len(logevent['Pump0']['Time_E'])}")
    print(f"VisualStimulate : {len(logevent['VisualStimulate']['Time_E'])}")
    msSyncTimes = len(logevent['MsSync']['Time_E']) / 20 / 60
    print(f"MsSync last for : {msSyncTimes:.2f}m")
    baslerSyncTimes = len(logevent['BaslerSync']['Time_E']) / 50 / 60
    print(f"BaslerSync last for : {baslerSyncTimes:.2f}m")
    print()


