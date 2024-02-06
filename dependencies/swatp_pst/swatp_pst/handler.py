import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from hydroeval import evaluator, nse, rmse, pbias
import numpy as np
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


def create_swatp_pst_con(
    prj_dir, swatp_wd, cal_start, cal_end, chans, time_step=None
    ):

    if time_step is None:
        time_step = 'day'

    col01 = [
        'prj_dir',
        'swatp_wd', 'cal_start', 'cal_end',
        'chs',
        'time_step',
        ]
    col02 = [
        prj_dir,
        swatp_wd, cal_start, cal_end, 
        chs,
        time_step,
        ]
    df = pd.DataFrame({'names': col01, 'vals': col02})

    main_opt_path = os.path.join(prj_dir, 'main_opt')
    if not os.path.isdir(main_opt_path):
        os.makedirs(main_opt_path)
    with open(os.path.join(prj_dir, 'main_opt', 'swatp_pst.con'), 'w', newline='') as f:
        f.write("# swatp_pst.con created by swatp_pst\n")
        df.to_csv(
            f, sep='\t',
            encoding='utf-8', 
            index=False, header=False)
    return df


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

        # copy files from opt_files folder
        for j in filesToCopy:
            if not os.path.isfile(os.path.join(main_opt_path, j)):
                shutil.copy2(os.path.join(opt_files_path, j), os.path.join(main_opt_path, j))
                print(" '{}' file copied ...".format(j) + colored(suffix, 'green'))
        # copy forward_run.py
        if not os.path.isfile(os.path.join(main_opt_path, 'forward_run.py')):
            shutil.copy2(os.path.join(foward_path, 'forward_run.py'), os.path.join(main_opt_path, 'forward_run.py'))
            print(" '{}' file copied ...".format('forward_run.py') + colored(suffix, 'green'))
        os.chdir(main_opt_path)       
    else:
        print("failed to create 'main_opt' folder, folder already exists ..." )
    os.chdir(main_opt_path)
    print(f"path to main_opt folder: {main_opt_path}")


class SWATp(object):
    def __init__(self, wd):
        os.chdir(wd)
        self.stdate, self.enddate, self.stdate_warmup = self.define_sim_period()

    def define_sim_period(self):
        df_time = self.read_time_sim()
        df_prt = self.read_print_prt()
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

    def read_time_sim(self):
        return pd.read_csv(
            "time.sim",
            sep=r'\s+',
            skiprows=1,
        )

    def read_print_prt(self):
        return pd.read_csv(
            "print.prt",
            sep=r'\s+',
            skiprows=1
        )

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

    def extract_mon_stf(self, chs, cali_start_day, cali_end_day):
        sim_stf_f = self.read_cha_morph_mon()
        start_day = self.stdate_warmup
        for i in chs:
            sim_stf_f = self.read_cha_morph_mon()
            sim_stf_f = sim_stf_f.loc[sim_stf_f["gis_id"] == i]
            sim_stf_f = sim_stf_f.drop(['gis_id'], axis=1)
            sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.flo_out), freq='ME')
            sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
            sim_stf_f.to_csv(
                'stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
            print('stf_{:03d}.txt file has been created...'.format(i))
        print('Finished ...')




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

        # copy files from opt_files folder
        for j in filesToCopy:
            if not os.path.isfile(os.path.join(main_opt_path, j)):
                shutil.copy2(os.path.join(opt_files_path, j), os.path.join(main_opt_path, j))
                print(" '{}' file copied ...".format(j) + colored(suffix, 'green'))
        # copy forward_run.py
        if not os.path.isfile(os.path.join(main_opt_path, 'forward_run.py')):
            shutil.copy2(os.path.join(foward_path, 'forward_run.py'), os.path.join(main_opt_path, 'forward_run.py'))
            print(" '{}' file copied ...".format('forward_run.py') + colored(suffix, 'green'))
        os.chdir(main_opt_path)       
    else:
        print("failed to create 'main_opt' folder, folder already exists ..." )
    os.chdir(main_opt_path)
    print(f"path to main_opt folder: {main_opt_path}")

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








if __name__ == '__main__':
    wd = "/Users/seonggyu.park/Documents/projects/tools/swatp-pest_wf/models/TxtInOut_Imsil_rye_rot_r1"
    # wd = "D:\\Projects\\Watersheds\\Koksilah\\analysis\\koksilah_swatmf\\SWAT-MODFLOW"

    m1 = SWATp(wd)
    cns =  [1]
    cali_start_day = "1/1/2013"
    cali_end_day = "12/31/2023"
    obd_file = "singi_obs_q1_colnam.csv"
    obd_colnam = "cha01"
    cha_ext_file = "stf_001.txt"

    m1.stf_obd_to_ins(cha_ext_file, obd_file, obd_colnam, cali_start_day, cali_end_day, time_step="month")

    # print(dff)