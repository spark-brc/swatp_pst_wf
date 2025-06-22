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
    # print('\n' + 35*'+ ')
    print('\n > '+ time + ' |  {} ...'.format(des))
    # print(35*'+ ' + '\n')

def execute_swatp():
    des = "running model"
    time_stamp(des)
    # pyemu.os_utils.run('APEX-MODFLOW3.exe >_s+m.stdout', cwd='.')
    pyemu.os_utils.run('rev61.0.1_64rel.exe', cwd='.')

def extract_stf_results(chs, cal_start, cal_end, tstep=None):
    # get time step
    if tstep is None:
        time_step == 'day'
    if time_step == 'month':
        des = "extracting monthly simulated streamflow"
        time_stamp(des)
        m1.extract_mon_stf(chs, cal_start, cal_end)
    elif time_step == 'day':
        des = "extracting daily simulated streamflow"
        time_stamp(des)
        m1.extract_day_stf(chs, cal_start, cal_end)

def extract_sim_waterlevel(chs, cal_start, cal_end, tstep=None):
    # get time step
    if tstep is None:
        time_step == 'day'
    if time_step == 'month':
        des = "extracting monthly simulated water level"
        time_stamp(des)
        m1.extract_sim_waterlevel(chs, cal_start, cal_end, tstep='month')
    elif time_step == 'day':
        des = "extracting daily simulated water level"
        time_stamp(des)
        m1.extract_sim_waterlevel(chs, cal_start, cal_end)


if __name__ == '__main__':
    os.chdir(wd)
    swatp_pst_con = pd.read_csv('swatp_pst.con', sep='\t', names=['names', 'vals'], index_col=0, comment="#")
    cal_start = swatp_pst_con.loc['cal_start', 'vals']
    cal_end = swatp_pst_con.loc['cal_end', 'vals']
    time_step = swatp_pst_con.loc['time_step','vals']
    # execute_swatp()

    if swatp_pst_con.loc['chs', 'vals'] != 'n':
        chs = swatp_pst_con.loc['chs','vals'].strip('][').split(', ')
        chs = [int(i) for i in chs]
        extract_stf_results(chs, cal_start, cal_end)

    if swatp_pst_con.loc['grids', 'vals'] != 'n':
        grids = swatp_pst_con.loc['grids','vals'].strip('][').split(', ')
        grids = [int(i) for i in grids]
        extract_sim_waterlevel(grids, cal_start, cal_end)

    if swatp_pst_con.loc['irr_cal', 'vals'] == 'activated':
        m1 = SWATp(wd)
        m1.get_mon_irr()    
    print(wd)







