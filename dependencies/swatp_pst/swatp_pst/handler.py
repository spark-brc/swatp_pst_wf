import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import datetime as dt
from datetime import datetime
import calendar
import shutil
from tqdm import tqdm
from termcolor import colored
from shutil import copyfile
from swatp_pst import analyzer
from warnings import simplefilter
import matplotlib.gridspec as gridspec

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
opt_files_path = os.path.join(
                    os.path.dirname(os.path.abspath( __file__ )),
                    'opt_files')
foward_path = os.path.dirname(os.path.abspath( __file__ ))
suffix = "OK"


def create_swatp_pst_con(
            prj_dir, swatp_wd, cal_start, cal_end, chs, 
            irr_cal=None,
            time_step=None
            ):
    if time_step is None:
        time_step = 'day'
    if irr_cal:
        irr_cal = "activated"
    col01 = [
        'prj_dir',
        'swatp_wd', 'cal_start', 'cal_end',
        'chs',
        'irr_cal',
        'time_step',
        ]
    col02 = [
        prj_dir,
        swatp_wd, cal_start, cal_end, 
        chs,
        irr_cal, 
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


# from cjfx
def file_name(path_, extension=True):
    if extension:
        fn = os.path.basename(path_)
    else:
        fn = os.path.basename(path_).split(".")[0]
    return(fn)

def read_from(filename, decode_codec = None, v=False):
    '''
    a function to read ascii files
    '''
    try:
        if not decode_codec is None: g = open(filename, 'rb')
        else: g = open(filename, 'r')
    except:
        print(
            "\t! error reading {0}, make sure the file exists".format(filename))
        return

    file_text = g.readlines()
    
    if not decode_codec is None: file_text = [line.decode(decode_codec) for line in file_text]

    if v:
        print("\t> read {0}".format(file_name(filename)))
    g.close
    return file_text

def get_file_size(file_path):
    return float(os.path.getsize(file_path))/1012

def error(text_):
    print("\t! {string_}".format(string_=text_))

def create_path(path_name, v=False):
    path_name = os.path.dirname(path_name)
    if path_name == '':
        path_name = './'
    if not os.path.isdir(path_name):
        os.makedirs(path_name)
        if v:
            print(f"\t> created path: {path_name}")
    return path_name

# from cjfx
def copy_file(filename, destination_path, delete_source=False, v = False, replace = True):
    '''
    a function to copy files
    '''
    if not replace:
        if exists(destination_path):
            if v:
                print(f"\t - file exists, skipping")
            return
    if not exists(filename):
        if not v:
            return
        print("\t> The file you want to copy does not exist")
        print(f"\t    - {filename}\n")
        ans = input("\t> Press  E then ENTER to Exit or C then ENTER to continue: ")
        counter = 0
        while (not ans.lower() == "c") and (not ans.lower() == "e"):
            ans = input("\t> Please, press E then ENTER to Exit or C then ENTER to continue: ")
            if counter > 2:
                print("\t! Learn to read instrunctions!!!!")
            counter += 1
        if ans.lower() == 'e': quit()
        if ans.lower() == 'c':
            write_to("log.txt", f"{filename}\n", mode='append')
            return
    if v:
        if delete_source:
            print(f"\t - [{get_file_size(filename)}] moving {filename} to \n\t\t{destination_path}")
        else:
            # print(f"\t - [{get_file_size(filename)}] copying {filename} to \n\t\t{destination_path}")
            sys.stdout.write('\rcopying ' + filename + '                        ')
            sys.stdout.flush()

    if not os.path.isdir(os.path.dirname(destination_path)):
        try:
            os.makedirs(os.path.dirname(destination_path))
        except:
            pass
    copyfile(filename, destination_path)
    if delete_source:
        try:
            os.remove(filename)
        except:
            error('coule not remove {fl}, make sure it is not in use'.format(fl=filename))

def write_to(filename, text_to_write, v=False, mode = "overwrite"):
    '''
    a function to write to file
    modes: overwrite/o; append/a
    '''
    try:
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
            if v:
                print("! the directory {0} has been created".format(
                    os.path.dirname(filename)))
    except:
        pass

    if (mode == "overwrite") or (mode == "o"):
        g = open(filename, 'w', encoding="utf-8")
    elif (mode == "append") or (mode == "a"):
        g = open(filename, 'a', encoding="utf-8")
    try:
        g.write(text_to_write)
        if v:
            print('\n\t> file saved to ' + filename)
    except PermissionError:
        print("\t> error writing to {0}, make sure the file is not open in another program".format(
            filename))
        response = input("\t> continue with the error? (Y/N): ")
        if response == "N" or response == "n":
            sys.exit()
    g.close

def exists(path_):
    if os.path.isdir(path_):
        return True
    if os.path.isfile(path_):
        return True
    return False

class SWATp(object):
    def __init__(self, working_dir):
        self.working_dir = working_dir
        os.chdir(self.working_dir)
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

    def read_hru_data(self):
        return pd.read_csv(
            "hru-data.hru",
            sep=r'\s+',
            skiprows=1
        )        

    def read_hru_con(self):
        return pd.read_csv(
            "hru.con",
            sep=r'\s+',
            skiprows=1
        )
    
    def read_hru_wb_mon(self):
        return pd.read_csv(
            "hru_wb_mon.txt",
            sep=r'\s+',
            skiprows=[0,2]
        )        

    def read_hru_wb_aa(self):
        return pd.read_csv(
            "hru_wb_aa.txt",
            sep=r'\s+',
            skiprows=[0,2]
        )      


    def read_hru_pw_yr(self):
        return pd.read_csv(
            "hru_pw_yr.txt",
            sep=r'\s+',
            skiprows=[0,2]
        )     


    def create_paddy_hru_id_database(self):
        hru_area = self.read_hru_con()
        hru_area = hru_area.loc[:, ["id", "area"]]
        hru_area.set_index('id', inplace=True)
        hru_paddy = self.read_hru_data()
        hru_paddy.dropna(subset=['surf_stor'], inplace=True)
        hru_paddy = hru_paddy[hru_paddy['surf_stor'].str.contains('paddy')]
        hru_paddy.set_index('id', inplace=True)
        hru_paddy = pd.concat([hru_paddy, hru_area], axis=1)
        hru_paddy['hruid'] = hru_paddy.index
        hru_paddy.dropna(subset=['surf_stor'], axis=0, inplace=True)
        # tot_area = hru_paddy.loc[:, "area"].sum()
        # hru_paddy["area_weighted"] =hru_paddy.loc[:, "area"]/ tot_area, 
        # print(hru_paddy)
        return hru_paddy


    def read_cha_morph_day(self, flo=None):
        if flo is None:
            flo = "flo_out"
        if flo == "flo_in":
            flo = "flo_in"
        return pd.read_csv(
            "channel_sdmorph_day.txt",
            sep=r'\s+',
            skiprows=[0,2],
            usecols=["gis_id", flo]
            )
    
    def read_cha_sd_day(self, flo=None):
        if flo is None:
            flo = "flo_out"
        if flo == "flo_in":
            flo = "flo_in"
        return pd.read_csv(
            "channel_sd_day.txt",
            sep=r'\s+',
            skiprows=[0,2],
            usecols=["gis_id", flo]
            )
    
    def read_cha_sd_mon(self):
        return pd.read_csv(
            "channel_sd_mon.txt",
            sep=r'\s+',
            skiprows=[0,2],
            usecols=["gis_id", "flo_out"]
            )


    def read_cha_morph_mon(self):
        return pd.read_csv(
            "channel_sdmorph_mon.txt",
            sep=r'\s+',
            skiprows=[0,2],
            usecols=["gis_id", "flo_out"]
            )
    
    def read_basin_wb_mon(self):
        return pd.read_csv(
            "basin_wb_mon.txt",
            sep=r'\s+',
            skiprows=[0,2]            
        )

    def read_basin_wb_yr(self):
        return pd.read_csv(
            "basin_wb_yr.txt",
            sep=r'\s+',
            skiprows=[0,2]            
        )   


    def read_basin_wb_aa(self):
        return pd.read_csv(
            "basin_wb_aa.txt",
            sep=r'\s+',
            header=None,
            skiprows=[0, 2]            
        )   

    def read_basin_wb_aa2(self) -> Series:
        """read basin wb aa file and make as series

        :return: s
        :rtype: Series
        """
        with open("basin_wb_aa.txt", "r") as f:
            data = f.readlines()
        firstline = data[1].strip().split()
        flen = len(firstline)
        secondline = data[3].strip().split()
        secondline = secondline[:flen]
        s = pd.Series(secondline, firstline)
        return s

    def read_basin_aqu_aa(self) -> Series:
        with open("basin_aqu_aa.txt", "r") as f:
            data = f.readlines()
        firstline = data[1].strip().split()
        flen = len(firstline)
        secondline = data[3].strip().split()
        secondline = secondline[:flen]
        s = pd.Series(secondline, firstline)
        return s        


    def read_basin_sed_aa(self) -> Series:
        """read basin wb aa file and make as series

        :return: s
        :rtype: Series
        """
        with open("basin_ls_aa.txt", "r") as f:
            data = f.readlines()
        firstline = data[1].strip().split()
        flen = len(firstline)
        secondline = data[3].strip().split()
        secondline = secondline[:flen]
        s = pd.Series(secondline, firstline)
        return s



    def read_cha_obd(self, obd_file):
        return pd.read_csv(
            obd_file,
            na_values=["", -999],
            index_col=0,
            parse_dates=True,
        )
    
    def read_pcp_data(self):
        return pd.read_csv(
            "channel_sd_mon.txt",
            sep=r'\s+',
            # skiprows=[0,2],
            usecols=["gis_id", "area", "precip", ]
            )
    
    def read_lu_wb_mon(self):
        return pd.read_csv(
            "lsunit_wb_mon.txt",
            sep=r'\s+',
            skiprows=[0,2]            
        )
    
    def read_lu_wb_aa(self):
        return pd.read_csv(
            "lsunit_wb_aa.txt",
            sep=r'\s+',
            skiprows=[0,2]            
        )
    
    def read_lu_wb_yr(self):
        return pd.read_csv(
            "lsunit_wb_yr.txt",
            sep=r'\s+',
            skiprows=[0,2]            
        )
    
    def read_ls_def(self):
        return pd.read_csv(
            "ls_unit.def",
            sep=r'\s+',
            skiprows=3,
            header=None,
            usecols=[1, 2],
            names=["name", "area"]
        )

    def read_crop_yld_aa(self):
        return pd.read_csv(
            "crop_yld_aa.txt",
            sep=r'\s+',
            skiprows=[0,1,2,3]            
        )

    def read_hru_lsu(self):
        return pd.read_csv(
                    "hru_lsu.csv",
                    usecols=[0, 1],
                    )
    
    def get_hru_lsu_df(self):
        df = self.read_hru_lsu()
        counts = df['runame'].value_counts()
        return counts

    def read_hru_con(self):
        return pd.read_csv(
                    "hru.con",
                    sep=r'\s+',
                    skiprows=[0]     
                    )
    
    def read_hru_data(self):
        return pd.read_csv(
                    "hru-data.hru",
                    sep=r'\s+',
                    skiprows=[0]     
                    )        

    def get_hru_area(self):
        df = self.read_hru_con()
        areas = df.loc[:, 'area'].tolist()
        return areas
    
    def get_landuse(self):
        ha = self.read_hru_con().loc[:, ["name", "area"]]
        ha.index = ha.name
        hd = self.read_hru_data().loc[:, ["name", "lu_mgt"]]
        hd.index = hd.name
        hd.fillna("barr_lum", inplace=True)
        df = pd.concat([ha, hd], axis=1)
        df.drop("name", axis=1, inplace=True)

        df = df.groupby('lu_mgt').sum()
        tot_area = df["area"].sum()
        df['perct'] = (df['area']/tot_area) * 100
        return df

    def read_basin_pw_mon(self):
        return pd.read_csv(
                        "basin_pw_mon.txt",
                        sep=r'\s+',
                        skiprows=[0,2]   
                        )
    
    def get_monthly_temps(self):
        df = self.read_basin_pw_mon()
        start_day = self.stdate_warmup
        df = df.loc[:, ["tmx", "tmn", "tmpav"]]
        df.index = pd.date_range(start=start_day, periods=len(df), freq="ME")
        df = df.groupby(df.index.month).mean()
        return df


    def read_basin_wb_mon(self):
        return pd.read_csv(
                        "basin_wb_mon.txt",
                        sep=r'\s+',
                        skiprows=[0,2]   
                        )

    def get_monthly_precip(self):
        df = self.read_basin_wb_mon()
        start_day = self.stdate_warmup
        df = df.loc[:, "precip"]
        df.index = pd.date_range(start=start_day, periods=len(df), freq="ME")
        df = df.groupby(df.index.month).mean()
        return df
   
    def monthly_weather(self):
        df = pd.concat([self.get_monthly_precip(), self.get_monthly_temps()], axis=1)
        return df
        
    def get_monthly_irr(self):
        df = self.read_basin_wb_mon()
        start_day = self.stdate_warmup
        df = df.loc[:, "irr"]
        df.index = pd.date_range(start=start_day, periods=len(df), freq="ME")
        df = df.groupby(df.index.month).mean()
        return df

    def monthly_weather_irr(self):
        tot_irr = self.get_mon_irr()
        tot_irr = tot_irr.loc[:, "tot_irr"].groupby(tot_irr.index.month).mean()
        df = pd.concat(
            [
                self.get_monthly_precip(), 
                self.get_monthly_temps(),
                tot_irr 
                ], axis=1)
        return df

    def read_basin_crop_yr(self, crop="rice"):
        df = pd.read_csv("basin_crop_yld_yr.txt", sep=r"\s+", index_col=False, skiprows=[0])
        df = df.loc[df["plant_name"]==crop]
        df.index = df["year"]
        return df
    
    def read_yield_obd(self, filenam):
        df = pd.read_csv(filenam, na_values=-999, index_col=0)
        return df

    def get_crop_yld_sim_obd(self, filenam, crop="rice"):
        df = pd.concat(
            [self.read_basin_crop_yr(crop).loc[:, "yld(t/ha)"], self.read_yield_obd(filenam)], axis=1
        )
        df.dropna(how='all', inplace=True)
        return df

    def extract_mon_stf(self, chs, cali_start_day, cali_end_day):
        sim_stf_f = self.read_cha_morph_mon()
        start_day = self.stdate_warmup
        for i in chs:
            # sim_stf_f = self.read_cha_morph_mon()
            sim_stf_ff = sim_stf_f.loc[sim_stf_f["gis_id"] == i]
            sim_stf_ff = sim_stf_ff.drop(['gis_id'], axis=1)
            sim_stf_ff.index = pd.date_range(start_day, periods=len(sim_stf_ff.flo_out), freq='ME')
            sim_stf_ff = sim_stf_ff[cali_start_day:cali_end_day]
            sim_stf_ff.to_csv(
                'stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
            print('stf_{:03d}.txt file has been created...'.format(i))
        print('Finished ...')


    def extract_day_stf(self, chs, cali_start_day, cali_end_day):
        sim_stf_f = self.read_cha_morph_day()
        start_day = self.stdate_warmup
        for i in chs:
            # sim_stf_f = self.read_cha_morph_mon()
            sim_stf_ff = sim_stf_f.loc[sim_stf_f["gis_id"] == i]
            sim_stf_ff = sim_stf_ff.drop(['gis_id'], axis=1)
            sim_stf_ff.index = pd.date_range(start_day, periods=len(sim_stf_ff.flo_out))
            sim_stf_ff = sim_stf_ff[cali_start_day:cali_end_day]
            sim_stf_ff.to_csv(
                'stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
            print(' >>> stf_{:03d}.txt file has been created...'.format(i))
        print(' > Finished ...\n')


    def extract_day_stf_albu(self, chs, cali_start_day, cali_end_day):
        sim_stf_f = self.read_cha_morph_day(flo="flo_in")
        start_day = self.stdate_warmup
        for i in chs:
            # sim_stf_f = self.read_cha_morph_mon()
            sim_stf_ff = sim_stf_f.loc[sim_stf_f["gis_id"] == i]
            sim_stf_ff = sim_stf_ff.drop(['gis_id'], axis=1)
            sim_stf_ff.index = pd.date_range(start_day, periods=len(sim_stf_ff.flo_in))
            sim_stf_ff = sim_stf_ff[cali_start_day:cali_end_day]
            sim_stf_ff.to_csv(
                'stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
            print(' >>> stf_{:03d}.txt file has been created...'.format(i))
        print(' > Finished ...\n')

    def extract_crop_aa(self, crops):
        sim_stf_f = self.read_crop_yld_aa()
        for i in crops:
            # sim_stf_f = self.read_cha_morph_mon()
            sim_stf_ff = sim_stf_f.loc[sim_stf_f["PLANTNM"] == i]
            sim_stf_ff = sim_stf_ff.drop(['jday', 'mon', 'day', 'yr'], axis=1)
            print(sim_stf_ff)
            sim_stf_ff.to_csv(
                f'crop_aa_{i}.txt', sep='\t', index=False, header=False,
                float_format='%.7e')
            print(f'crop_aa_{i}.txt file has been created...')
        print('Finished ...')


    def get_mon_irr(self):
        paddy_df = pd.DataFrame()
        paddy_hru_id = self.create_paddy_hru_id_database()
        df = self.read_hru_wb_mon()
        for hruid in paddy_hru_id.loc[:, "hruid"]:
            paddy_df[f"hru_{hruid}"] = df.loc[df["unit"]==hruid, "irr"].values
        paddy_df.index = pd.date_range(
            self.stdate_warmup, periods=len(paddy_df), freq="ME")
        # print(paddy_df)
        
        paddy_ids = [int(f"{pid[4:]}") for pid in paddy_df.columns]
        paddy_hru_id = paddy_hru_id.query('hruid in @paddy_ids')
        tot_area = paddy_hru_id.loc[:, "area"].sum()
        paddy_hru_id["area_weighted"] =paddy_hru_id.loc[:, "area"]/ tot_area

        # make total based size weighted average
        irr_ratio = pd.DataFrame()
        for hruid in paddy_hru_id.loc[:, "hruid"]:
            # if f"hru_{hruid}" in paddy_df.columns:
            irr_ratio[f"irr_ratio_{hruid}"] = (
                paddy_df.loc[:, f"hru_{hruid}"] * 
                paddy_hru_id.loc[paddy_hru_id["hruid"]==hruid, 'area_weighted'].values
            )
        paddy_df['tot_irr'] = irr_ratio.sum(axis=1)
        paddy_df.to_csv("irr_paddy_wb.csv")
        paddy_df.loc[:, 'tot_irr'].to_csv(
                "tot_irr_paddy.txt", sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e'
                        )
        print(' > tot_irr_paddy.txt file has been created...\n')
        return paddy_df

    def get_lu_mon(self, field, stdate=None, eddate=None):
        suffix = "passed"
        lu_df = pd.DataFrame()
        lu_mon_df = self.read_lu_wb_mon()
        luids = lu_mon_df.name.unique()
        for luid in luids:
            lu_df[f"{luid}"] = lu_mon_df.loc[lu_mon_df["name"]==luid, field].values
        lu_df.index = pd.date_range(
            self.stdate_warmup, periods=len(lu_df), freq="ME")
        if stdate is not None:
            dff = lu_df[stdate:eddate].astype(float)
        else:
            dff = lu_df.astype(float)
        mbig_df = dff.groupby(dff.index.month).mean().T
        mbig_df['lsuid'] = [int(i[3:]) for i in mbig_df.index]
        mbig_df.to_csv(f"lsu_{field}_mon_wb.csv", index=False)
        print(f' > lsu_{field}_mon_wb.csv file has been created... '+ colored(suffix, 'green'))
        print(os.getcwd())
        return mbig_df

    def get_lu_wb_aa(self):
        suffix = "OK"
        columns = ["name", "wateryld", "perc", "et", "sw_ave"]
        lu_aa_df = self.read_lu_wb_aa().loc[:, columns]
        lu_aa_df['lsuid'] = [int(i[3:]) for i in lu_aa_df["name"]]
        lu_aa_df.drop("name", axis=1, inplace=True)
        lu_aa_df.to_csv(f"lsu_aa_wb.csv", index=False)
        print(f' > lsu_aa_wb.csv file has been created... '+ colored(suffix, 'green'))


    def get_hru_mon(self, field, stdate=None, eddate=None):
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        hru_df = pd.DataFrame()
        hru_mon_df = self.read_hru_wb_mon()
        hruids = hru_mon_df.name.unique()
        for hruid in hruids:
            hru_df[f"{hruid}"] = hru_mon_df.loc[hru_mon_df["name"]==hruid, field].values
        hru_df.index = pd.date_range(
            self.stdate_warmup, periods=len(hru_df), freq="ME")
        if stdate is not None:
            dff = hru_df[stdate:eddate].astype(float)
        else:
            dff = hru_df.astype(float)
        mbig_df = dff.groupby(dff.index.month).mean().T

        mbig_df['hruid'] = [int(i[3:]) for i in mbig_df.index]
        mbig_df.to_csv(f"hru_{field}_mon_wb.csv", index=False)
        return mbig_df

    def get_lu_hf_wb_(self):
        lu_hf = self.read_lu_wb_yr()
        # filter >=2017
        lu_hf = lu_hf.loc[lu_hf['yr']>=2013]
        cols = [
                "name", "yr",
                "precip", "surq_gen", "latq", "wateryld", "perc", 
                "et", "sw_ave", "irr", "surq_runon", "latq_runon",
                # "surq_cha"
                ]
        lu_hf = lu_hf[cols]
        lu_hf = lu_hf.groupby(lu_hf['name']).mean()
        lu_area = self.read_ls_def()
        lu_area.set_index('name', inplace=True)
        lu_hf = pd.concat([lu_hf, lu_area], axis=1)
        hill_df = lu_hf.loc[lu_hf.index.str.strip().str[-1]=='2']
        hill_totarea = hill_df['area'].sum()
        hill_df['weight_area'] = hill_df['area'] / hill_totarea
        hill_wt = pd.DataFrame()
        for col in hill_df.columns[1:-2]:
            hill_wt[f'{col}'] = hill_df[f'{col}'] * hill_df['weight_area']
            # hill_wt = pd.concat([hill_wt, hill_df[f'{col}'] * hill_df['weight_area']], axis=1)
        hill_sum = hill_wt.sum(axis=0)
        fdp_df = lu_hf.loc[lu_hf.index.str.strip().str[-1]=='1']
        fdp_totarea = fdp_df['area'].sum()
        fdp_df['weight_area'] = fdp_df['area'] / fdp_totarea
        fdp_wt = pd.DataFrame()
        for col in fdp_df.columns[1:-2]:
            fdp_wt[f'{col}'] = fdp_df[f'{col}'] * fdp_df['weight_area']
            # fdp_wt = pd.concat([fdp_wt, fdp_df[f'{col}'] * fdp_df['weight_area']], axis=1)
        fdp_sum = fdp_wt.sum(axis=0)

        tot_df = pd.concat([hill_sum, fdp_sum], axis=1)
        tot_df.columns = ["hillslope", "floodplain"]
        tot_df.to_csv("hf.csv")
        print(os.getcwd())

        print(tot_df)
        print(lu_hf)
        return tot_df


    def get_lu_hf_wb(self):
        lu_hf = self.read_lu_wb_aa()
        # filter >=2017
        lu_hf = lu_hf.loc[lu_hf['yr']>=2017]
        cols = [
                "name", "yr",
                "precip", "surq_gen", "latq", "wateryld", "perc", 
                "et", "sw_ave", "irr", "surq_runon", "latq_runon",
                # "surq_cha"
                ]
        lu_hf = lu_hf[cols]
        lu_hf = lu_hf.groupby(lu_hf['name']).mean()
        lu_area = self.read_ls_def()
        lu_area.set_index('name', inplace=True)
        lu_hf = pd.concat([lu_hf, lu_area], axis=1)
        hill_df = lu_hf.loc[lu_hf.index.str.strip().str[-1]=='2']
        hill_totarea = hill_df['area'].sum()
        hill_df['weight_area'] = hill_df['area'] / hill_totarea
        hill_wt = pd.DataFrame()
        for col in hill_df.columns[1:-2]:
            hill_wt[f'{col}'] = hill_df[f'{col}'] * hill_df['weight_area']
            # hill_wt = pd.concat([hill_wt, hill_df[f'{col}'] * hill_df['weight_area']], axis=1)
        hill_sum = hill_wt.sum(axis=0)
        fdp_df = lu_hf.loc[lu_hf.index.str.strip().str[-1]=='1']
        fdp_totarea = fdp_df['area'].sum()
        fdp_df['weight_area'] = fdp_df['area'] / fdp_totarea
        fdp_wt = pd.DataFrame()
        for col in fdp_df.columns[1:-2]:
            fdp_wt[f'{col}'] = fdp_df[f'{col}'] * fdp_df['weight_area']
            # fdp_wt = pd.concat([fdp_wt, fdp_df[f'{col}'] * fdp_df['weight_area']], axis=1)
        fdp_sum = fdp_wt.sum(axis=0)
        tot_df = pd.concat([hill_sum, fdp_sum], axis=1)
        tot_df.columns = ["hillslope", "floodplain"]
        print(tot_df)
        """
        tot_df.to_csv("hf.csv")
        print(os.getcwd())
        return tot_df
        """
        print(lu_hf)

    def get_landuse_wb_aa(
            self, field:str
            ) -> Series:
        """_summary_

        :param field: water balance type
        :type field: str
        :return: weighted water balace
        :rtype: Series
        """
        dff = pd.DataFrame()
        df = self.read_hru_wb_aa()
        df.set_index('name', inplace=True)
        hru_area = self.read_hru_con()
        hru_area.set_index('name', inplace=True)
        hru_data = self.read_hru_data()
        hru_data.set_index('name', inplace=True)
        hru_data["lu_type"] = [str(i)[:-4] if i !=np.nan else i for i in hru_data["lu_mgt"]]
        # hru_data = hru_data["lu_type"]
        dff = pd.concat(
            [hru_data["lu_type"], 
             hru_area["area"],
             df[f"{field}"]
             ], axis=1)
        # weight by area
        tot_area = dff.loc[:, "area"].sum()
        dff["area_weighted"] =dff.loc[:, "area"]/ tot_area
        dff[f"{field}_weigthed"] = dff[field] * dff["area_weighted"]
        fdf = dff.groupby(dff["lu_type"]).sum()
        return fdf.iloc[:, -1]
        

class CliScenario:
    def __init__(self, working_dir):
        self.working_dir = working_dir

    def read_lu_wb_csv(self, scn_dir, field):
        df = pd.read_csv(
            os.path.join(self.working_dir, scn_dir, f"lsu_{field}_mon_wb.csv"))
        df[f"MJ_{scn_dir}"] = (df["5"] + df["6"])/2
        df[f"JD_{scn_dir}"] = (df["1"] + df["12"])/2
        df = df[[f"MJ_{scn_dir}", f"JD_{scn_dir}", "lsuid"]]

        # print(df)
        return df

    def get_lu_wb_mon_scns(self, scn_dirs, field):
        dff = pd.DataFrame()
        for sd in scn_dirs:
            df = self.read_lu_wb_csv(sd, field)
            dff = pd.concat([dff, df], axis=1)
        dff = dff.loc[:,~dff.columns.duplicated()].copy()
        # basecols = ["hist", "ssp245"]
        # base_cols = dff.loc[:, dff.columns.str.contains('and'.join(basecols), case=False)]
        mjdf = dff.loc[:, dff.columns.str.contains('MJ', case=False)]
        mjbase = mjdf.loc[:, mjdf.columns.str.contains('hist', case=False)]
        mjscn = mjdf.loc[:,~mjdf.columns.str.contains('hist', case=False)] 
        mjpct = pd.DataFrame()
        for i in mjscn.columns:
            mjpct[f"pct_{i}"] = ((mjscn[i] - mjbase.iloc[:, 0]) /mjbase.iloc[:, 0]) * 100
        # mjpct = pd.concat([mjpct, mjbase.iloc[:, 0], dff["lsuid"]], axis=1)
        jddf = dff.loc[:, dff.columns.str.contains('JD', case=False)]
        jdbase = jddf.loc[:, jddf.columns.str.contains('hist', case=False)]
        jdscn = jddf.loc[:,~jddf.columns.str.contains('hist', case=False)] 

        jdpct = pd.DataFrame()
        for i in jdscn.columns:
            jdpct[f"pct_{i}"] = ((jdscn[i] - jdbase.iloc[:, 0]) /jdbase.iloc[:, 0]) * 100

        totdf = pd.DataFrame()
        totdf = pd.concat([dff["lsuid"], mjbase.iloc[:, 0], mjpct, jdbase.iloc[:, 0], jdpct], axis=1)
        totdf.replace([np.inf, -np.inf], np.nan, inplace=True)

        totdf.to_csv(os.path.join(self.working_dir, f"lsu_{field}_wb_mon_scn.csv"), index=False)
        print(f' > lsu_{field}_wb_mon_scn.csv file has been created... '+ colored(suffix, 'green'))

    def read_lu_wb_aa_csv(self, scn_dir):
        df = pd.read_csv(
            os.path.join(self.working_dir, scn_dir, f"lsu_aa_wb.csv"))
        df.columns = [f"{scn_dir}_{i}" if i != "lsuid" else i for i in df.columns]
        # print(df)
        return df
    
    def get_lu_wb_aa_scns(self, scn_dirs):
        dff = pd.DataFrame()
        for sd in scn_dirs:
            df = self.read_lu_wb_aa_csv(sd)
            dff = pd.concat([dff, df], axis=1)
        dff = dff.loc[:,~dff.columns.duplicated()].copy()
        fields = ["wateryld", "perc", "et", "sw_ave"]
        for f in fields:
            totdf = dff.loc[:, dff.columns.str.contains(f, case=False)]
            basedf = totdf.loc[:, totdf.columns.str.contains('hist', case=False)]
            scndf = totdf.loc[:,~totdf.columns.str.contains('hist', case=False)] 
            pctdf = pd.DataFrame()
            pctdf['lsuid'] = dff.loc[:, "lsuid"]
            pctdf['base'] = basedf.iloc[:, 0]

            # for j in scndf.columns:
            #     pctdf[f"pct_{j}"] = ((scndf[j] - basedf.iloc[:, 0]) /basedf.iloc[:, 0]) * 100
            #     pctdf.to_csv(os.path.join(self.working_dir, f"lsu_{f}_wb_aa_scn.csv"), index=False)
            #     print(f' > lsu_{f}_wb_aa_scn.csv file has been created... '+ colored(suffix, 'green'))
            for j in scndf.columns:
                pctdf[f"pct_{j}"] = ((scndf[j] - basedf.iloc[:, 0]) /basedf.iloc[:, 0]) * 100
            pctdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            pctdf.to_csv(os.path.join(self.working_dir, f"lsu_{f}_wb_aa_scn.csv"), float_format= '%.2f', index=False)
            print(f' > lsu_{f}_wb_aa_scn.csv file has been created... '+ colored(suffix, 'green'))

    def get_basin_wb_aa_scns(
            self, 
            scn_dirs: list,
        ) -> DataFrame:
        # waterbalance info
        wdf = pd.DataFrame()
        for sd in scn_dirs:
            smodel = SWATp(os.path.join(self.working_dir, sd))
            s = smodel.read_basin_wb_aa2()
            s.name = f"{sd}"
            wdf = pd.concat([wdf, s], axis=1)
        # sediment
        sdf = pd.DataFrame()
        for sd in scn_dirs:
            smodel = SWATp(os.path.join(self.working_dir, sd))
            s = smodel.read_basin_sed_aa()
            s.name = f"{sd}"
            sdf = pd.concat([sdf, s], axis=1)   
        # sediment
        aqdf = pd.DataFrame()
        for sd in scn_dirs:
            smodel = SWATp(os.path.join(self.working_dir, sd))
            s = smodel.read_basin_aqu_aa()
            s.name = f"{sd}"
            aqdf = pd.concat([aqdf, s], axis=1)  
            
        wdf = pd.concat([wdf, sdf, aqdf], axis=0)
        wdf.to_csv(os.path.join(self.working_dir, "basin_wb_aa_scns.csv"))
        return wdf


    def get_streamdischarge_scns(self, scn_dirs, chaid, timestep="day"):        
        dff = pd.DataFrame()
        for scn_dir in scn_dirs:
            m1 = SWATp(os.path.join(self.working_dir, scn_dir))
            if timestep == "day":
                freq = "D"
                df = m1.read_cha_morph_day()
            elif timestep == "month":
                freq = "ME"
                df = m1.read_cha_morph_mon()
            start_day = m1.stdate_warmup
            df = df.loc[df["gis_id"] == chaid]
            df = df.drop(['gis_id'], axis=1)
            df.index = pd.date_range(start_day, periods=len(df.flo_out), freq=freq)
            df.columns = [f"{scn_dir}"]
            dff = pd.concat([dff, df], axis=1)
            print(f"  ... {scn_dir} ... " + colored(suffix, 'green'))
        return dff

    def extract_stf_day_scns(self, scn_dirs, chaid, timestep="day"):        
        dff = pd.DataFrame()
        for scn_dir in scn_dirs:
            m1 = SWATp(os.path.join(self.working_dir, scn_dir))
            if timestep == "day":
                freq = "D"
                df = m1.read_cha_sd_day()
            elif timestep == "month":
                freq = "ME"
                df = m1.read_cha_sd_mon()
            start_day = m1.stdate_warmup
            df = df.loc[df["gis_id"] == chaid]
            df = df.drop(['gis_id'], axis=1)
            df.index = pd.date_range(start_day, periods=len(df.flo_out), freq=freq)
            df.columns = [f"{scn_dir}"]
            dff = pd.concat([dff, df], axis=1)
            print(f"  ... {scn_dir} ... " + colored(suffix, 'green'))
        dff.to_csv(
            os.path.join(self.working_dir,'stf_{:03d}_day_scns.csv'.format(chaid)),
            encoding='utf-8', index=True, header=True,
            float_format='%.5e', na_rep='-999')
        print(' >>> stf_{:03d}_day_scns.csv file has been created...'.format(chaid))
        print(' > Finished ...\n')
        return dff

    def read_stf_scns(self, chaid, timestep="day"):
        if timestep == "day":
            cha_extract_file = f"stf_{chaid:03d}_day_scns.csv"
        stf_sim = pd.read_csv(
                            os.path.join(self.working_dir, cha_extract_file),
                            index_col=0,
                            parse_dates=True,
                            na_values=-999)
        return stf_sim

    def get_landuse_wb_aa_scns(self, scn_dirs, field) -> DataFrame:
        dff = pd.DataFrame()
        for scn_dir in scn_dirs:
            m1 = SWATp(os.path.join(self.working_dir, scn_dir))
            df = m1.get_landuse_wb_aa(field)
            df.name = scn_dir
            dff = pd.concat([dff, df], axis=1)
        dff.to_csv(os.path.join(self.working_dir, f"landuse_{field}_aa_scns.csv"))
        return dff

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


class Paddy(SWATp):
    def __init__(self, wd, *args, **kwargs):
        super().__init__(wd, *args, **kwargs)
        self.wd = wd
        os.chdir(self.wd)
        self.stdate, self.enddate, self.stdate_warmup = self.define_sim_period()
    
    def read_print_prt(self):
        return pd.read_csv(
            "print.prt",
            sep=r'\s+',
            skiprows=1
        )

    def read_time_sim(self):
        return pd.read_csv(
            "time.sim",
            sep=r'\s+',
            skiprows=1,
        )
    
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

    def read_paddy_daily(self, hruid=None):
        if hruid is None:
            hruid = 1 
        df = pd.read_csv("paddy_daily.csv", index_col=False)
        df = df.rename(columns=lambda x: x.strip())
        df = df.loc[df["HRU"]==1]
        df.index = pd.date_range(self.stdate_warmup, periods=len(df))
        return df

    def read_basin_pw_day(self, hruid=None):
        if hruid is None:
            hruid =1
        df = pd.read_csv("basin_pw_day.txt", sep=r"\s+", index_col=False, skiprows=[0, 2])
        df = df.rename(columns=lambda x: x.strip())
        df = df.loc[df["gis_id"]==1]
        df.index = pd.date_range(self.stdate_warmup, periods=len(df))
        return df

    def read_yield_obd(self):
        inf = "YIELD & PRODUCTION - DISTRICT DATA_csir request.xlsx"
        years, yields = [], []
        for i in range(2013, 2023):
            df = pd.read_excel(inf, sheet_name=str(i), skiprows=1, usecols=[1, 3])
            df.dropna(inplace=True)
            # df = df.loc[df["DISTRICTS"]=="Tolon/Kumbungu"]
            df = df[df["DISTRICTS"].str.contains('Kumbungu')]
            # df['Credit-Rating'].str.contains('Fair')
            years.append(i)
            yields.append(df.loc[:, "RICE"].values[0])
        dff = pd.Series(index=years, data=yields, name='obd_yield')
        dff.index = pd.date_range(f"1/1/{2013}", periods=len(dff), freq="YE")
        return dff

    def read_lsunit_wb_yr(self, hruid=None):
        if hruid is None:
            hruid =1
        df = pd.read_csv("lsunit_wb_yr.txt", sep=r"\s+", index_col=False, skiprows=[0, 2])
        df = df.rename(columns=lambda x: x.strip())
        df = df.loc[df["unit"]==1]
        df.index = pd.date_range(self.stdate_warmup, periods=len(df), freq="YE")
        return df
    
    def read_pcp_obd(self):
        inf = "pcp_year_obd.csv"
        df = pd.read_csv(inf, parse_dates=True, index_col=0)
        return df

    def create_backup(self, overwrite=False):
        filesToCopy = [
            "file.cio",
            "hru-data.hru",
            "hydrology.wet",
            "hydrology.hyd",
            "wetland.wet",
            "initial.res",
            "irr.ops",
            "landuse.lum",
            "plant.ini"
            ]
        suffix = ' passed'
        # print(" > Creating 'backup' folder in working directory ...",  end='\r', flush=True)
        print(" > Creating 'backup' folder in working directory ...")
        backup_path = os.path.join(os.getcwd(), 'backup')
        if not os.path.isdir(backup_path):
            os.makedirs(backup_path)
        for j in filesToCopy:
            if not os.path.isfile(os.path.join(backup_path, j)):
                shutil.copy2(os.path.join(os.getcwd(), j), os.path.join(backup_path, j))
                print(" >>> '{}' file copied ...".format(j) + colored(suffix, 'green'))
        print(" > Creating 'backup' folder in working directory ..." + colored(suffix, 'green'))

    def conv_hrudata(self, lumlist=None):
        if lumlist is None: # landcode
            lumlist = ["rice"]
        with open(os.path.join(self.wd, 'backup', 'hru-data.hru'), "r") as f:
            data = f.readlines()
            ndigits = len(str(data[-1].split()[0]))
            for ll in lumlist:
                c = 0
                for line in data:
                    if (
                        (len(line.split()) >=6) and 
                        (line.split()[5] != "null") and 
                        (line.split()[5].startswith(ll))
                    ):
                        new_line = self.replace_line_hrudata(line, ndigits)
                        data[c] = new_line
                    c += 1
        with open(os.path.join(self.wd, "hru-data.hru"), "w") as wf:
            wf.writelines(data)
        new_file = os.path.join(self.wd, 'hru-data.hru')
        print(
            f" {'>'*3} {os.path.basename(new_file)}" + 
            " file is overwritten successfully!"
            )

    def replace_line_hrudata(self, line, nd):
        parts = line.split()
        new_line = (
            f'{int(parts[0]):8d}'+ f'{parts[1]:>9s}'+ f'{parts[2]:>27s}'+
            f'{parts[3]:>18s}'+ f'{parts[4]:>18s}'+ f"{'rice_paddy_lum':>18s}"+
            f'{parts[6]:>18s}' + f"{f'paddy{int(parts[0][-4:]):>0{nd}d}':>18s}" + 
            f'{parts[8]:>18s}' + f'{parts[9]:>18s}'
            "\n"
        )
        return new_line
    
    def conv_wetlandwet(self):
        with open('hru-data.hru', "r") as f:
            data = f.readlines()
            paddy_objs = []
            for line in data:
                if len(line.split()) >=7 and line.split()[7].startswith("paddy"):
                    paddy_objs.append(line.split()[7])

        with open(os.path.join(self.wd, 'backup',"wetland.wet"), "r") as fw:
            data = fw.readlines()
            ndigits = len(str(data[-1].split()[0]))
            stid = int(data[-1].split()[0]) + 1
            for paddy_obj in paddy_objs:
                new_line = (
                    f'{int(stid):8d}' 
                    f'  {paddy_obj:<16s}' 
                    f"{'high_init':>18s}" 
                    f"{'paddy':>18s}" 
                    # f"{paddy_obj:>18s}" 
                    f"{'weir':>18s}" 
                    f"{'sedwet1':>18s}" 
                    f"{'nutwet1':>18s}\n" 
                )
                data.append(new_line)
                stid += 1

        with open("wetland.wet", "w") as wf:
            wf.writelines(data)
    # print(paddy_objs)

        '''
        with open(os.path.join(self.wd, 'backup', 'wetland.wet'), "r") as f:
            data = f.readlines()
            ndigits = len(str(data[-1].split()[0]))
            for ll in lumlist:
                c = 0
                for line in data:
                    if line.split()[5] != "null" and line.split()[5].startswith(ll):
                        new_line = self.replace_line(line, ndigits)
                        data[c] = new_line
                    c += 1
        '''

    def replace_line_wetlandwet(self, line, nd):
        parts = line.split()
        new_line = (
            f'{int(parts[0]):8d}'+ f'{parts[1]:>9s}'+ f'{parts[2]:>27s}'+
            f'{parts[3]:>18s}'+ f'{parts[4]:>18s}'+ f"{'rice_paddy_lum':>18s}"+
            f'{parts[6]:>18s}' + f"{f'paddy{int(parts[0][-4:]):>0{nd}d}':>18s}" + 
            f'{parts[8]:>18s}' + f'{parts[9]:>18s}'
            "\n"
        )
        return new_line

    def conv_filecio(self):
        with open(os.path.join(self.wd, 'backup', 'file.cio'), "r") as f:
            data = f.readlines()
            c = 0
            modi = "notyet"
            for line in data:
                if line.split()[0] == "reservoir" and not "weir.res" in line.split():
                    modi = "y"
                    nc = len(line.split())
                    ridx = line.split().index("null")
                    newlist = line.split()
                    newlist[ridx] = "weir.res"
                    newline = []
                    for i in newlist:
                        newline.append(f'{i:<18s}')
                    newline.append("\n")
                    newline ="".join(newline)
                    data[c] = newline
                c += 1
        if modi == "y":    
            with open(os.path.join(self.wd, "file.cio"), "w") as wf:
                wf.writelines(data)
            new_file = os.path.join(self.wd, 'file.cio')
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is overwritten successfully!"
                )
        else:
            new_file = os.path.join(self.wd, 'file.cio')
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is not overwritten!"
                )

    def conv_initialres(self):
        with open(os.path.join(self.wd, 'backup',"initial.res"), "r") as fw:
            data = fw.readlines()
            fc = [line.split()[0] for line in data]
        modi = 'n'
        if "low_init" not in fc:
            modi = 'y'
            lowinit_line = (
                f"{'low_init':<16s}"+ f"{'low_init':>18s}"+ f"{'no_ini':>18s}"+ f"{'no_ini':>18s}"+
                f"{'null':>18s}"+ f"{'null':>18s}"+ f"{'null':>18s}"
                "\n"                
            )
            data.append(lowinit_line)
        if "high_init" not in fc:
            modi = 'y'
            highinit_line = (
                f"{'high_init':<16s}"+ f"{'high_init':>18s}"+ f"{'low_ini':>18s}"+ f"{'low_ini':>18s}"+
                f"{'null':>18s}"+ f"{'null':>18s}"+ f"{'null':>18s}"
                "\n"                
            )
            data.append(highinit_line)
        if modi == "y":    
            with open("initial.res", "w") as wf:
                wf.writelines(data)
            new_file = os.path.join(self.wd, 'initial.res')
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is overwritten successfully!"
                )
        else:
            new_file = os.path.join(self.wd, 'initial.res')
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is not overwritten!"
                )

    def conv_irrops(self):
        with open(os.path.join(self.wd, 'backup',"irr.ops"), "r") as fw:
            data = fw.readlines()
            fc = [line.split()[0] for line in data if line !='\n']
        print(fc)
        modi = 'n'
        if "ponding90" not in fc:
            modi = 'y'
            ponding90_line = (
                f"{'ponding90':<16s}"+ f"{90:>14.5f}"+ f"{1:>14.5f}"+ f"{0:>14.5f}"+
                f"{60:>14.5f}"+ f"{0:>14.5f}"+ f"{0:>14.5f}" + f"{0:>14.5f}"
                "\n"                
            )
            data.append(ponding90_line)
        if "ponding_off" not in fc:
            modi = 'y'
            ponding_off_line = (
                f"{'ponding_off':<16s}"+ f"{0:>14.5f}"+ f"{1:>14.5f}"+ f"{0.1:>14.5f}"+
                f"{0:>14.5f}"+ f"{0:>14.5f}"+ f"{0:>14.5f}" + f"{0:>14.5f}"
                "\n"                
            )
            data.append(ponding_off_line)
        if modi == "y":    
            with open("irr.ops", "w") as wf:
                wf.writelines(data)
            new_file = os.path.join(self.wd, "irr.ops")
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is overwritten successfully!"
                )
        else:
            new_file = os.path.join(self.wd, "irr.ops")
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is not overwritten!"
                )

    def get_paddy_objs(self):
        with open('hru-data.hru', "r") as f:
            data = f.readlines()
            paddy_objs = []
            for line in data:
                if len(line.split()) >=7 and line.split()[7].startswith("paddy"):
                    paddy_objs.append([line.split()[1], line.split()[7]])
            paddy_objs = np.array(paddy_objs)
        return paddy_objs

    def conv_hydwet(self):
        with open('hru-data.hru', "r") as f:
            data = f.readlines()
            paddy_objs = []
            for line in data:
                if len(line.split()) >=7 and line.split()[7].startswith("paddy"):
                    paddy_objs.append(line.split()[7])


        with open(os.path.join(self.wd, 'backup',"hydrology.wet"), "r") as fw:
            data = fw.readlines()
            # ndigits = len(str(data[-1].split()[0]))
            # stid = int(data[-1].split()[0]) + 1
            for paddy_obj in paddy_objs:
                new_line = (
                    # f'{int(stid):8d}' + 
                    f'{paddy_obj:<16s}' 
                    f"{1:>14.5f}" 
                    f"{180:>14.5f}" 
                    f"{1:>14.5f}" 
                    f"{180:>14.5f}" 
                    f"{0.1:>14.5f}" 
                    f"{0.8:>14.5f}" 
                    f"{1:>14.5f}" 
                    f"{1:>14.5f}" 
                    f"{1:>14.5f}" 
                    f"{0.5:>14.5f}\n" 
                )
                data.append(new_line)
                # stid += 1

        with open("hydrology.wet", "w") as wf:
            wf.writelines(data)
    # print(paddy_objs)

        '''
        with open(os.path.join(self.wd, 'backup', 'wetland.wet'), "r") as f:
            data = f.readlines()
            ndigits = len(str(data[-1].split()[0]))
            for ll in lumlist:
                c = 0
                for line in data:
                    if line.split()[5] != "null" and line.split()[5].startswith(ll):
                        new_line = self.replace_line(line, ndigits)
                        data[c] = new_line
                    c += 1
        '''

    def conv_landlum(self):
        with open(os.path.join(self.wd, 'backup',"landuse.lum"), "r") as fw:
            data = fw.readlines()
            fc = [line.split()[0] for line in data if line !='\n']
        print(fc[2:])
        modi = 'n'
        if "rice_paddy_lum" not in fc:
            newlist = [
                'rice_paddy_lum', 'null', 'rice120_comm', 'paddy', 'legr_strow_p', 'ter_1-2_sodout', 'null', 'null', 'chisplow_nores',
                ] + ['null']*5
            newline = []
            for i in newlist:
                if i == 'rice_paddy_lum':
                    newline.append(f'{i:<20s}')
                elif i == 'paddy':
                    newline.append(f'{i:>43s}')
                else:
                    newline.append(f'{i:>18s}')
            newline.append("\n")
            newline ="".join(newline)            
            data.append(newline)
            modi = 'y'
        if modi == "y":    
            with open("landuse.lum", "w") as wf:
                wf.writelines(data)
            new_file = os.path.join(self.wd, "landuse.lum")
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is overwritten successfully!"
                )
        else:
            new_file = os.path.join(self.wd, "landuse.lum")
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is not overwritten!"
                )

    def conv_plantin(self):
        inf = "plant.ini"
        with open(os.path.join(self.wd, 'backup',inf), "r") as fw:
            data = fw.readlines()
            fc = [line.split()[0] for line in data if line !='\n']
        print(fc[2:])
        modi = 'n'
        if "rice120_comm" not in fc:
            newlist = [
                'rice120_comm', 1, 1]
            newlist2 = [
                'rice120', 'n', 0, 0, 0, 0, 0, 10000
                ]           
            newline = []
            for i in newlist:
                if i == 'rice120_comm':
                    newline.append(f'{i:<16s}')
                else:
                    newline.append(f'{i:>10d}')
            newline.append("\n")
            newline ="".join(newline)
            newline2 = []
            for i in newlist2:
                if i == 'rice120':
                    newline2.append(f'{i:>44s}')
                elif i == "n" or i == "y":
                    newline2.append(f'{i:>14s}')
                else:
                    newline2.append(f'{i:>14.5f}')
            newline2.append("\n")
            newline2 ="".join(newline2)
            data.append(newline)
            data.append(newline2)
            modi = 'y'
        if modi == "y":    
            with open(inf, "w") as wf:
                wf.writelines(data)
            new_file = os.path.join(self.wd, inf)
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is overwritten successfully!"
                )
        else:
            new_file = os.path.join(self.wd, inf)
            print(
                f" {'>'*3} {os.path.basename(new_file)}" + 
                " file is not overwritten!"
                )

    def conv_hyd_perco(self, perco=None):
        if perco is None:
            perco = 0.0001
        # get paddy hru
        with open('hru-data.hru', "r") as f:
            data = f.readlines()
            paddy_objs = []
            for line in data:
                if len(line.split()) >=7 and line.split()[7].startswith("paddy"):
                    paddy_objs.append(line.split()[3])

        with open(os.path.join(self.wd, 'backup', 'hydrology.hyd'), "r") as f:
            data = f.readlines()
            c = 0
            for line in data:
                if line.split()[0] in paddy_objs:
                    new_line = self.replace_line_hyd(line, perco)
                    data[c] = new_line
                c += 1
        with open(os.path.join(self.wd, "hydrology.hyd"), "w") as wf:
            wf.writelines(data)
        new_file = os.path.join(self.wd, 'hydrology.hyd')
        print(
            f" {'>'*3} {os.path.basename(new_file)}" + 
            " file is overwritten successfully!"
            )

    def replace_line_hyd(self, line, perco):
        parts = line.split()
        newline = []
        for i in range(len(parts)):
            if i == 0:
                newline.append(f'{parts[i]:<16s}')
            elif i == 10:
                newline.append(f'{perco:>14.5f}')
            else:
                newline.append(f"{float(parts[i]):>14.5f}" )
        newline.append("\n")
        newline ="".join(newline)            
        return newline


    def copy_nfiles_paddy(self):
        suffix = ' passed'
        cfiles = ['weir.res', 'puddle.ops', 'swatplus.exe']
        for cfile in cfiles:
            if not os.path.isfile(cfile):
                shutil.copy2(os.path.join(opt_files_path, cfile), os.path.join(self.wd, cfile))
                print(" >>> '{}' file copied ...".format(cfile) + colored(suffix, 'green'))
            else:
                print(" >>> '{}' file already exist ...".format(cfile) + colored(suffix, 'green'))


    def conv_paddy(self):
        self.create_backup()
        self.conv_hrudata()
        self.conv_wetlandwet()
        self.conv_filecio()
        self.conv_initialres()
        self.conv_irrops()
        self.conv_hydwet()
        self.conv_landlum()
        self.conv_plantin()
        self.conv_hyd_perco()
        self.copy_nfiles_paddy()

    def filter_paddy(self, pid=None):
        if pid is None:
            suffix = "it will read all data"
            print(" > provide paddy id ... or " + colored(suffix, 'red'))
            df = pd.read_csv("paddy_daily.csv")
            print(df.head())
        else:
            df = pd.read_csv("paddy_daily.csv")
            df.columns = [i.strip() for i in df.columns]
            df = df.loc[df["HRU"]==pid]
            df.to_csv(f"paddy_daily_{pid:d}.csv", index=False)
            print(df.head())
        return df

    def viz_pp(self):
        print('test')


    #NOTE: read txt file, original tmp input file
    def generate_heatunit_org(wd, inf, cropBHU, month, day):
        df = pd.read_csv(os.path.join(wd, inf), skiprows=1, names=['tmax', 'tmin'], na_values=-999)
        stdate = read_from(os.path.join(wd, inf))[0]
        df.index = pd.date_range(start=stdate, periods=len(df))
        df['tmean'] = (df['tmin'] + df['tmax'])/2
        df["HU"] = df['tmean'] - cropBHU
        df.loc[df['HU'] < 0, 'HU'] = 0
        df[f"PHU{cropBHU}"] = df.groupby(df.index.year)["HU"].cumsum()
        
        phu0 = df.loc[(df.index.month==month) & (df.index.day==day)] 
        tphu0 = df.loc[(df.index.month==12) & (df.index.day==31)] 

        dff = pd.DataFrame(index=df.index.year.unique(), columns=["PHU0", "TPHU0"])
        dff["PHU0"] = phu0.loc[:, "PHU0"].values
        dff["TPHU0"] = tphu0.loc[:, "PHU0"].values
        dff["FPHU0"] = dff["PHU0"] / dff["TPHU0"]

        dff.to_csv(os.path.join(wd, 'test.csv')) 
        return dff


    def heatunit_days(self, inf, month, wd=None, cropBHU=None):
        if wd is None:
            wd = os.getcwd()
        if cropBHU is None:
            cropBHU = 0
        df = pd.read_csv(
            os.path.join(wd, inf), 
            skiprows=3, names=['yr', 'doy', 'tmax', 'tmin'], na_values=-999,
            sep=r'\s+')
        df['date'] = pd.to_datetime(df['yr'] * 1000 + df['doy'], format='%Y%j')
        df.index = df['date']
        df.drop('date', axis=1, inplace=True)
        df = df.loc[(df.index.month==month)]
        days = df.index.day.unique().tolist()
        return days

    def generate_heatunit(self, inf, month, day, cropBHU=None, wd=None):
        if wd is None:
            wd = os.getcwd()
        if cropBHU is None:
            cropBHU = 0
        df = pd.read_csv(
            os.path.join(wd, inf), 
            skiprows=3, names=['yr', 'doy', 'tmax', 'tmin'], na_values=-999,
            sep=r'\s+')
        df['date'] = pd.to_datetime(df['yr'] * 1000 + df['doy'], format='%Y%j')
        df.index = df['date']
        df.drop('date', axis=1, inplace=True)
        df['tmean'] = (df['tmin'] + df['tmax'])/2
        df["HU"] = df['tmean'] - cropBHU
        df.loc[df['HU'] < 0, 'HU'] = 0
        df[f"PHU{cropBHU}"] = df.groupby(df.index.year)["HU"].cumsum()
        
        if day in df.index.day:
            phu0 = df.loc[(df.index.month==month) & (df.index.day==day)]
        tphu0 = df.loc[(df.index.month==12) & (df.index.day==31)] 
        tphu0 = tphu0[tphu0["yr"].isin(phu0.loc[:, 'yr'])]
        
        # yrs = phu0.index.tolist()
        # print(tphu0[tphu0["yr"].isin(phu0.loc[:, 'yr'])])
        # tphu0 = tphu0.loc[['1984-02-29', '1988-02-29', '1992-02-29', '1996-02-29']]
        # print(tphu0.index)
        # # print(phu0)

        dff = pd.DataFrame(index=phu0.index.year.unique(), columns=["PHU0", "TPHU0"])
        dff["PHU0"] = phu0.loc[:, "PHU0"].values
        dff["TPHU0"] = tphu0.loc[:, "PHU0"].values
        dff["FPHU0"] = dff["PHU0"] / dff["TPHU0"]
        dff.to_csv(os.path.join(wd, 'test.csv')) 
        return dff

    def get_paddy_stress_df(self):
        paddy_objs = self.get_paddy_objs()[:, 0] # get hru name
        columns = ['name', 'strsw', 'strsa', 'strstmp', 'strsn', 'strsp', 'strss']
        df = self.read_hru_pw_yr()
        df = df[columns]
        df = df[df['name'].isin(paddy_objs)]
        df = df.groupby(df['name']).mean()
        return df
    
    def get_crop_yld(self):
        paddy_objs = self.get_paddy_objs()[:, 0] # get hru name
        columns = ['yr', 'name', "yield"]
        df = self.read_hru_pw_yr()
        df = df[columns]
        df = df[df['name'].isin(paddy_objs)]
        df = df.loc[df["yr"]==2020]
        return df


class Executes:

    def combined():
        # NOTE: PADDY
        wd =  "d:\\Projects\\Watersheds\\Ghana\\Analysis\\botanga\\prj01\\Scenarios\\Default\\TxtInOut_rice_f"
        m1 = Paddy(wd)

        '''
        '''
        df = m1.read_paddy_daily()
        cols = ["Precip", "Irrig", "Seep", "ET", "PET", 'WeirH', 'Wtrdep', 'WeirQ','LAI']
        df = df.loc[:,  cols]
        df = df["1/1/2019":"12/31/2020"]
        print(df)
        analyzer.Paddy(wd).plot_paddy_daily(df)

        dfs = m1.read_lsunit_wb_yr()
        dfs = dfs.loc[:,  "precip"]
        dfo = m1.read_pcp_obd()
        print(dfs)

        dfs = pd.concat([dfs, dfo], axis=1)
        analyzer.Paddy(wd).plot_prep(dfs)
        # print(analyzer.Paddy(wd).stdate)

        dfy = m1.read_basin_pw_day()
        dfy = dfy.loc[:,  "yield"].resample('YE').sum() * 0.001
        dfyo = m1.read_yield_obd()
        dfy = pd.concat([dfy, dfyo], axis=1)

        print(dfy)
        analyzer.Paddy(wd).plot_yield(dfy)
    

        # NOTE: extract crop
        m1.extract_crop_aa(['rice_dwn'])

        # NOTE: filter paddy
        # m1.filter_paddy(2899)
        # df = m1.get_hru_lsu_df()

        sitenam = "Dawhenya"
        # print(type(df))

        df = m1.get_hru_area()
        # df = [x for x in df if x < 300]
        fig, axes = plt.subplots(1, 2, figsize=(3, 5))
        axes[0] = analyzer.SWATp.violin_hru_lsu(axes[0], df, sitenam)

        plt.show()


        # NOTE: get paddy stress bar
        df = m1.get_paddy_stress_df()
        mv1 = analyzer.SWATp(m1.wd)
        for c in df.columns:
            mv1.plot_stress(df, stress=c, h=2)
        print(df)

        # NOTE: filter paddy
        m1.conv_hrudata()
        m1.conv_hyd_perco(perco=0.1)
        m1 = SWATp(wd)
        fields = ["wateryld", "perc", "et", "sw_ave", "latq_runon"]
        for fd in fields:
            m1.get_lu_mon(fd)
            print(fd)
        m2 = SWATp(wd)
        fields = ["wateryld", "perc", "et", "sw_ave", "latq_runon"]
        for fd in fields:
            print(m2.get_hru_mon(fd))




        # NOTE: get landuse hf waterbalance
        wd = "D:\\jj\\opt_3rd\\calibrated_model_v02"
        # wd = "D:\\Projects\\Watersheds\\Koksilah\\analysis\\koksilah_swatmf\\SWAT-MODFLOW"

        m1 = SWATp(wd)
        # cns =  [1]
        # cali_start_day = "1/1/2013"
        # cali_end_day = "12/31/2023"
        # obd_file = "singi_obs_q1_colnam.csv"
        # obd_colnam = "cha01"
        # cha_ext_file = "stf_001.txt"
        # fields = ["wateryld", "perc", "et", "sw_ave", "latq_runon"]
        # for fd in fields:
        #     m1.get_lu_mon(fd, stdate="1/1/2017", eddate="12/31/2023")
        #     print(fd)

        m1.get_lu_hf_wb()


    def landuse_wb_figure():
        # wd = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
        wd = "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default"
        m1 = CliScenario(wd)

        scns = []
        for sc in ["245", "585"]:
            for cset in ["hist", "near", "mid", "far"]:
                scns.append(f"ssp{sc}_{cset}")

        fields = ["wateryld", "perc", "et", "sw_ave"]
        # ldtypes = ["crwo", "fomi", "rice_paddy", "urbn"] # dawhenya
        ldtypes = ["tfoe", "tagr", "rice140", "tswi"]
        colors = [f"C{i}" for i in range(8)]


        
        wyld = m1.get_landuse_wb_aa_scns(scns, "wateryld").loc[ldtypes]
        etdf = m1.get_landuse_wb_aa_scns(scns, "et").loc[ldtypes]
        percdf = m1.get_landuse_wb_aa_scns(scns, "perc").loc[ldtypes]
        swdf = m1.get_landuse_wb_aa_scns(scns, "sw_ave").loc[ldtypes]

        totdf = [etdf, wyld, swdf, percdf]

        # dff.drop("ssp585_hist", axis=1, inplace=True)
        # dff.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
        for ld in ldtypes:
            print(ld)
            f, axes = plt.subplots(4, 1, figsize=(4, 5), sharex=True)
            for i , ax in enumerate(axes.flat):
                cdf = totdf[i].copy()
                cdf.drop("ssp585_hist", axis=1, inplace=True)
                cdf.rename({'ssp245_hist': 'Historical'}, axis=1, inplace=True)
                xlabels = [x for x in cdf.columns]
                bar_container = ax.bar(cdf.columns, cdf.loc[ld], color=colors, alpha=0.5)


                pcs = Executes.percent_change(cdf, ld)
                print(pcs)
                pcs = [""] + [f"{p:.1f}" for p in pcs]

                # bar_container = ax.bar(fruit_names, fruit_counts)
                # ax.set(ylabel='pints sold', title='Gelato sales by flavor', ylim=(0, 8000))
                label_colors = []
                for pc in pcs:
                    try:
                        val = float(pc)
                        if val > 50:
                            colr = "#ff0000"
                        elif val <=50 and val >=25:
                            colr = "#ff8b00"
                        elif val < 25 and val > -25:
                            colr = "g"
                        elif val <= -25 and val >= -50:
                            colr = "#3d89ff"
                        elif val < -50:
                            colr = "b"
                        else:
                            colr = "k"
                    except ValueError as verr:
                        val = pc
                        colr = "k"
                        pass 
                    label_colors.append(colr)
                
                print(label_colors)

                labels = ax.bar_label(
                        bar_container, labels=pcs,
                        # padding=-10, 
                        color=colr, 
                        fontsize=10)
                for (label, lcolor) in zip(labels, label_colors):
                    label.set_color(lcolor)
                ax.margins(y=0.3)
                ax.tick_params(axis='both', labelsize=12)
                ax.set_xticks([i for i in range(len(xlabels))])
                ax.set_xticklabels(xlabels, rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(wd, f'{ld}_wb_scns.png'), dpi=300, bbox_inches="tight")

            plt.show()




    def landuse_wb_figure_not():
        # wd = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
        wd = "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default"
        m1 = CliScenario(wd)

        scns = []
        for sc in ["245", "585"]:
            for cset in ["hist", "near", "mid", "far"]:
                scns.append(f"ssp{sc}_{cset}")

        fields = ["wateryld", "perc", "et", "sw_ave"]
        ldtypes = ["crwo", "fomi", "rice_paddy", "urbn"]
        colors = [f"C{i}" for i in range(8)]


        
        wyld = m1.get_landuse_wb_aa_scns(scns, "wateryld").loc[ldtypes]
        etdf = m1.get_landuse_wb_aa_scns(scns, "et").loc[ldtypes]
        percdf = m1.get_landuse_wb_aa_scns(scns, "perc").loc[ldtypes]
        swdf = m1.get_landuse_wb_aa_scns(scns, "sw_ave").loc[ldtypes]

        totdf = [etdf, wyld, swdf, percdf]

        # dff.drop("ssp585_hist", axis=1, inplace=True)
        # dff.rename({'ssp245_hist': 'base'}, axis=1, inplace=True)
        for ld in ldtypes:
            print(ld)
            f, axes = plt.subplots(4, 1, figsize=(4, 5), sharex=True)
            for i , ax in enumerate(axes.flat):
                cdf = totdf[i].copy()
                cdf.drop("ssp585_hist", axis=1, inplace=True)
                cdf.rename({'ssp245_hist': 'Historical'}, axis=1, inplace=True)
                xlabels = [x for x in cdf.columns]


                for col in range(7):

                    bar_container = ax.bar(cdf.columns, cdf.loc[ld], color=colors, alpha=0.5)


                    pcs = Executes.percent_change(cdf, ld)
                    print(pcs)
                    pcs = [""] + [f"{p:.1f}" for p in pcs]

                    # bar_container = ax.bar(fruit_names, fruit_counts)
                    # ax.set(ylabel='pints sold', title='Gelato sales by flavor', ylim=(0, 8000))
                    label_colors = []
                    for pc in pcs:
                        try:
                            val = float(pc)
                            if val > 50:
                                colr = "#ff0000"
                            elif val <=50 and val >=25:
                                colr = "#ff8b00"
                            elif val < 25 and val > -25:
                                colr = "g"
                            elif val <= -25 and val >= -50:
                                colr = "#3d89ff"
                            elif val < -50:
                                colr = "#b"
                            else:
                                colr = "k"
                        except ValueError as verr:
                            val = pc
                            colr = "k"
                            pass 
                        label_colors.append(colr)
                    print(label_colors)


                    # for bar, color, pc in zip(bar_container, label_colors, pcs):
                    ax.bar_label(
                            bar_container, labels=pc,
                            # padding=-10, 
                            color=colr, 
                            fontsize=10)


                ax.margins(y=0.3)
                ax.tick_params(axis='both', labelsize=12)
                ax.set_xticks([i for i in range(len(xlabels))])
                ax.set_xticklabels(xlabels, rotation=90)
            plt.tight_layout()  
            plt.show()



    def percent_change(cdf, ld):
        base = cdf.loc[ld, cdf.columns[0]]
        s2nf = cdf.loc[ld, cdf.columns[1]]
        s2mf = cdf.loc[ld, cdf.columns[2]]
        s2ff = cdf.loc[ld, cdf.columns[3]]
        s5nf = cdf.loc[ld, cdf.columns[4]]
        s5mf = cdf.loc[ld, cdf.columns[5]]
        s5ff = cdf.loc[ld, cdf.columns[6]]

        percs = []
        for i in range(1, 7):
            pc =  (
                (cdf.loc[ld, cdf.columns[i]] - cdf.loc[ld, cdf.columns[0]])/
                cdf.loc[ld, cdf.columns[0]])*100
            percs.append(pc)
        return percs

        # fig = plt.figure(figsize=(10, 10))
        # outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

        # for i in range(4):
        #     inner = gridspec.GridSpecFromSubplotSpec(4, 1,
        #                     subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        #     for j in range(4):
        #         ax = plt.Subplot(fig, inner[j])
        #         cdf = totdf[j]
        #         ax.bar(cdf.columns, cdf.loc[ldtypes[i]])
        #         # t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
        #         # t.set_ha('center')
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         fig.add_subplot(ax)

        # plt.show()


    def example():
        colors = ["k", "r", "g"]

        fig, axe = plt.subplots()
        bars = axe.bar(["A", "B", "C"], height=[4, 6, 2], color=colors, label="Data")
        labels = axe.bar_label(bars, fontsize=10, fontweight=700, fmt="%.1f")
        for (label, color) in zip(labels, colors):
            label.set_color(color)
        axe.axhline(4, linestyle="-.", color="black", label="Limit")
        axe.legend()
        plt.show()



if __name__ == '__main__':


    # wd = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default"
    # m1 = CliScenario(wd)

    # scns = []
    # for sc in ["245", "585"]:
    #     for cset in ["hist", "near", "mid", "far"]:
    #         scns.append(f"ssp{sc}_{cset}")

    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    # m1 = SWATp(wd)
    # print(m1.get_landuse_wb_aa("wateryld"))

    # print(hru_area)

    Executes.landuse_wb_figure()

    
    


