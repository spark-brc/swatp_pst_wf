import pandas as pd
from swatp_pst.handler import SWATp
import calendar
import numpy as np
import os
import pyemu
from swatp_pst import handler


class gwFlow:

    def __init__(self, wd):
        os.chdir(wd)

    # NOTE: loop and add info in dictionary
    def gw_input_to_tpl(self):
        gwInFile = "gwflow.input"
        tpl_file = gwInFile + ".tpl"
        with open(gwInFile, "r") as inf:
            data = inf.readlines()
        # print(data)
        parinfo = {}
        for i , line in enumerate(data):
            if line.strip().lower().startswith("aquifer hydraulic conductivity"):
                parinfo['hc'] = [i+1, int(data[i+1].strip())]
            if line.strip().lower().startswith("aquifer specific yield"):
                parinfo['sy'] = [i+1, int(data[i+1].strip())]
            if line.strip().lower().startswith("streambed hydraulic conductivity"):
                parinfo['strbed_hc'] = [i+1, int(data[i+1].strip())]     
            if line.strip().lower().startswith("streambed thickness"):
                parinfo['strbed_tk'] = [i+1, int(data[i+1].strip())]

        for parnam in parinfo.keys():
            paridx = parinfo[parnam][0] + 1
            parlen = parinfo[parnam][1]

            for i in range(paridx, paridx+parlen):
                line  = data[i]
                newline = self.replace_line(parnam, line)
                data[i] = newline
        
        with open(tpl_file, "w") as wf:
            wf.writelines(data)
        print('done')

    def replace_line(self, parnam, line):
        parts = line.split()
        new_line = (
            f'{int(parts[0]):<4d}' + f' ~ {parnam}{int(parts[0]):03d} ~ ' + "\n"
            )
        return new_line


        # with open(tpl_file, 'w') as f:
        #     f.write("ptf ~\n")

    # def replace_line(self):



if __name__ == '__main__':
    # wd = "C:\\Mac\\Home\\Documents\\projects\\TxtInOut"
    wd = "/Users/seonggyu.park/Documents/projects/TxtInOut"

    # wd = "c:\\Mac\\Home\\Downloads\\Tutorial 2024\\Tutorial 2024\\QSWAT+ Application\\Complete_Tordera"
    # wd = "/Users/seonggyu.park/Downloads/Tutorial 2024/Tutorial 2024/QSWAT+ Application/Complete_Tordera/Scenarios/Default/TxtInOut"
    # wd = 
    stdate =  "01/01/2000"
    eddate = "12/31/2001"

    m1 = gwFlow(wd)
    m1.gw_input_to_tpl()
    mout = handler.SWATp(wd)
    mout.extract_mon_stf([106], stdate, eddate)
    mout.extract_sim_waterlevel(stdate, eddate)
    
    # print(os.getcwd())