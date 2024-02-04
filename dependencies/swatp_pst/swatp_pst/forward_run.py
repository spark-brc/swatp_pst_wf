# let's go~~~
import os
from datetime import datetime
import pyemu
import pandas as pd
import sys
import subprocess

# path = "D:/spark/gits/swatmf"
# sys.path.insert(1, path)

from swatp_pst.handler import SWATp

wd = os.getcwd()
os.chdir(wd)
m1 = SWATp(wd)

def time_stamp(des):
    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
    print('\n' + 35*'+ ')
    print(time + ' |  {} ...'.format(des))
    print(35*'+ ' + '\n')

def execute_swatp():
    des = "running model"
    time_stamp(des)
    # pyemu.os_utils.run('APEX-MODFLOW3.exe >_s+m.stdout', cwd='.')
    pyemu.os_utils.run('swatplus.exe', cwd='.')

def extract_stf_results(subs, cal_start, cal_end, tstep=None):
    # get time step
    if tstep is None:
        time_step == 'month'
    if time_step == 'month':
        des = "simulation successfully completed | extracting monthly simulated streamflow"
        time_stamp(des)
        m1.extract_mon_stf(subs, sim_start, warmup, cal_start, cal_end)
    elif time_step == 'day':
        des = "simulation successfully completed | extracting daily simulated streamflow"
        time_stamp(des)
        m1.extract_day_stf(subs, sim_start, warmup, cal_start, cal_end)





if __name__ == '__main__':
    os.chdir(wd)
    swatmf_con = pd.read_csv('swatp_pst.con', sep='\t', names=['names', 'vals'], index_col=0, comment="#")
    # get default vals
    # wd = swatmf_con.loc['wd', 'vals']
    sim_start = swatmf_con.loc['sim_start', 'vals']
    warmup = swatmf_con.loc['warm-up', 'vals']
    cal_start = swatmf_con.loc['cal_start', 'vals']
    cal_end = swatmf_con.loc['cal_end', 'vals']
    cha_act = swatmf_con.loc['subs','vals']
    grid_act = swatmf_con.loc['grids','vals']
    riv_parm = swatmf_con.loc['riv_parm', 'vals']
    baseflow_act = swatmf_con.loc['baseflow', 'vals']
    time_step = swatmf_con.loc['time_step','vals']
    pp_act = swatmf_con.loc['pp_included', 'vals']

    
    execute_swatp()
    # extract sims
    # if swatmf_con.loc['cha_file', 'vals'] != 'n' and swatmf_con.loc['fdc', 'vals'] != 'n':
    # if swatmf_con.loc['subs', 'vals'] != 'n':
    #     subs = swatmf_con.loc['subs','vals'].strip('][').split(', ')
    #     subs = [int(i) for i in subs]
    extract_stf_results(subs, sim_start, warmup, cal_start, cal_end)
    print(wd)





