import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from hydroeval import evaluator, nse, rmse, pbias
import numpy as np
import math
import datetime as dt
from datetime import datetime
import calendar
import shutil
from tqdm import tqdm
from termcolor import colored

opt_files_path = os.path.join(
                    os.path.dirname(os.path.abspath( __file__ )),
                    'opt_files')
foward_path = os.path.dirname(os.path.abspath( __file__ ))



def init_setup(prj_dir, swatp_wd):
    filesToCopy = [
        "i64pwtadj1.exe",
        "pestpp-glm.exe",
        "pestpp-ies.exe",
        "pestpp-opt.exe",
        "pestpp-sen.exe",
        ]
    suffix = ' passed'
    print(" Creating 'main_opt' folder in working directory ...",  end='\r', flush=True)

    main_opt_path = os.path.join(prj_dir, 'main_opt')

    if not os.path.isdir(main_opt_path):
        os.makedirs(main_opt_path)
    filelist = [f for f in os.listdir(swatp_wd) if os.path.isfile(os.path.join(swatp_wd, f))]
    for i in tqdm(filelist):
        # print(i)
        # if os.path.getsize(os.path.join(swatwd, i)) != 0:
        shutil.copy2(os.path.join(swatp_wd, i), main_opt_path)
    print(" Creating 'main_opt' folder ..." + colored(suffix, 'green'))

    # # create backup
    # print(" Creating 'backup' folder ...",  end='\r', flush=True)
    # if not os.path.isdir(os.path.join(main_opt_path, 'backup')):
    #     os.makedirs(os.path.join(main_opt_path, 'backup'))
    #     filelist = [f for f in os.listdir(swatwd) if os.path.isfile(os.path.join(swatwd, f))]
        
    #     # filelist =  os.listdir(swatwd)
    #     for i in tqdm(filelist):
    #         # print(i)
    #         # if os.path.getsize(os.path.join(swatwd, i)) != 0:
    #         shutil.copy2(os.path.join(swatwd, i), os.path.join(main_opt_path, 'backup'))
    # print(" Creating 'backup' folder ..." + colored(suffix, 'green'))

    # # create echo
    # print(" Creating 'echo' folder ...",  end='\r', flush=True)
    # if not os.path.isdir(os.path.join(main_opt_path, 'echo')):
    #     os.makedirs(os.path.join(main_opt_path, 'echo'))
    # print(" Creating 'echo' folder ..." + colored(suffix, 'green'))
    # # create sufi2
    # print(" Creating 'sufi2.in' folder ...",  end='\r', flush=True)
    # if not os.path.isdir(os.path.join(main_opt_path, 'sufi2.in')):
    #     os.makedirs(os.path.join(main_opt_path, 'sufi2.in'))
    # print(" Creating 'sufi2.in' folder ..."  + colored(suffix, 'green'))

    for j in filesToCopy:
        if not os.path.isfile(os.path.join(main_opt_path, j)):
            shutil.copy2(os.path.join(opt_files_path, j), os.path.join(main_opt_path, j))
            print(" '{}' file copied ...".format(j) + colored(suffix, 'green'))
    if not os.path.isfile(os.path.join(main_opt_path, 'forward_run.py')):
        shutil.copy2(os.path.join(foward_path, 'forward_run.py'), os.path.join(main_opt_path, 'forward_run.py'))
        print(" '{}' file copied ...".format('forward_run.py') + colored(suffix, 'green'))
    os.chdir(main_opt_path)       


def read_time_sim():
    return pd.read_csv(
        "time.sim",
        sep=r'\s+',
        skiprows=1,
    )

def read_print_prt():
    return pd.read_csv(
        "print.prt",
        sep=r'\s+',
        skiprows=1
    )

def define_sim_period():
    df_time = read_time_sim()
    df_prt = read_print_prt()
    skipyear = int(df_prt.loc[0, "nyskip"])
    yrc_start = int(df_time.loc[0, "yrc_start"])
    yrc_st_warmup = yrc_start + skipyear
    yrc_end = int(df_time.loc[0, "yrc_end"])
    start_day = int(df_time.loc[0, "day_start"])
    end_day = int(df_time.loc[0, "day_end"])
    stdate = dt.datetime(yrc_start, 1, 1) + dt.timedelta(start_day - 1)
    eddate = dt.datetime(yrc_end, 1, 1) + dt.timedelta(end_day - 1)
    stdate_warmup = dt.datetime(yrc_st_warmup, 1, 1) + dt.timedelta(start_day - 1)
    # eddate_warmup = dt.datetime(yrc_end_warmup, 1, 1) + dt.timedelta(FCendday - 1)
    
    startDate = stdate.strftime("%m/%d/%Y")
    endDate = eddate.strftime("%m/%d/%Y")
    startDate_warmup = stdate_warmup.strftime("%m/%d/%Y")
    # endDate_warmup = eddate_warmup.strftime("%m/%d/%Y")
    return startDate, endDate, startDate_warmup


# class SWATpIn(object):

# NOTE: swatp output handler
class SWATpOut(object):

    def __init__(self, wd):
        os.chdir(wd)
        if os.path.isfile("file.cio"):
            self.stdate, self.enddate, self.stdate_warmup = define_sim_period()

    def read_cha_morph_mon(self):
        return pd.read_csv(
            "channel_sdmorph_mon.txt",
            sep=r'\s+',
            skiprows=[0,2],
            usecols=["gis_id", "flo_out"]
            )
    
    def read_cha_obd(self, obd_file):
        return pd.read_csv(
            obd_file,
            na_values=["", -999],
            index_col=0,
            parse_dates=True,
        )

    # def extract_mon_stf(self, channels, cali_start_day, cali_end_day):
    #     sim_stf_f = self.read_cha_morph_mon()
    #     start_day = self.stdate
    #     for i in channels:
    #         sim_stf_f = self.read_cha_morph_mon()
    #         sim_stf_f = sim_stf_f.loc[sim_stf_f["gis_id"] == i]
    #         sim_stf_f = sim_stf_f.drop(['gis_id'], axis=1)
    #         sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.flo_out), freq='ME')
    #         sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
    #         sim_stf_f.to_csv(
    #             'stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False,
    #             float_format='%.7e')
    #         print('stf_{:03d}.txt file has been created...'.format(i))
    #     print('Finished ...')

    # def stf_obd_to_ins(self, cha_extract_file, obd_file, col_name, cal_start, cal_end, time_step=None):
    #     """extract a simulated streamflow from the output.rch file,
    #         store it in each channel file.

    #     Args:
    #         - rch_file (`str`): the path and name of the existing output file
    #         - channels (`list`): channel number in a list, e.g. [9, 60]
    #         - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
    #         - end_day ('str'): simulation end day e.g. '12/31/2000'
    #         - time_step (`str`): day, month, year

    #     Example:
    #         pest_utils.extract_month_stf('path', [9, 60], '1/1/1993', '12/31/2000')
    #     """ 
    #     if time_step is None:
    #         time_step = 'day'
    #         stfobd_file = 'stf_day.obd.csv'
    #     if time_step == 'month':
    #         stfobd_file = 'stf_mon.obd.csv'
    #     stf_obd = self.read_cha_obd(obd_file)
    #     stf_obd = get_last_day_of_month(stf_obd)
    #     stf_obd = stf_obd[cal_start:cal_end]
    #     stf_sim = pd.read_csv(
    #                         cha_extract_file,
    #                         sep=r'\s+',
    #                         names=["date", "stf_sim"],
    #                         index_col=0,
    #                         parse_dates=True)
    #     result = pd.concat([stf_obd, stf_sim], axis=1)
    #     result['tdate'] = pd.to_datetime(result.index)
    #     result['month'] = result['tdate'].dt.month
    #     result['year'] = result['tdate'].dt.year
    #     result['day'] = result['tdate'].dt.day
    #     if time_step == 'day':
    #         result['ins'] = (
    #                         'l1 w !{}_'.format(col_name) + result["year"].map(str) +
    #                         result["month"].map('{:02d}'.format) +
    #                         result["day"].map('{:02d}'.format) + '!'
    #                         )
    #     elif time_step == 'month':
    #         result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    #     else:
    #         print('are you performing a yearly calibration?')
    #     result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])
    #     with open(cha_extract_file+'.ins', "w", newline='') as f:
    #         f.write("pif ~" + "\n")
    #         result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    #     print('{}.ins file has been created...'.format(cha_extract_file))
    #     return result['{}_ins'.format(col_name)]


def get_last_day_of_month(df):
    for i in range(len(df)):
        res = calendar.monthrange(df.index[i].year, df.index[i].month)
        day = res[1]
        df.loc[df.index[i], "new_index"]= f"{df.index[i].year}-{df.index[i].month}-{day}"
    df.loc[:, "new_index"] = pd.to_datetime(df["new_index"])
    df.set_index(df["new_index"], inplace=True)
    df.drop("new_index", axis=1, inplace=True)
    df.index.rename("date", inplace=True)
    return df







if __name__ == '__main__':
    wd = "/Users/seonggyu.park/Documents/projects/tools/swatp-pest_wf/models/TxtInOut_Imsil_rye_rot_r1"
    # wd = "D:\\Projects\\Watersheds\\Koksilah\\analysis\\koksilah_swatmf\\SWAT-MODFLOW"

    m1 = SWATpOut(wd)
    cns =  [1]
    cali_start_day = "1/1/2013"
    cali_end_day = "12/31/2023"
    obd_file = "singi_obs_q1_colnam.csv"
    obd_colnam = "cha01"
    cha_ext_file = "stf_001.txt"

    m1.stf_obd_to_ins(cha_ext_file, obd_file, obd_colnam, cali_start_day, cali_end_day, time_step="month")

    # print(dff)