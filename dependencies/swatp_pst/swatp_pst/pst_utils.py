import pandas as pd
from swatp_pst.handler import SWATp
import calendar
import numpy as np
import os
import pyemu


class PstUtil(SWATp):
    def __init__(self, wd):
        super().__init__(wd)

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

    def stf_obd_to_ins(self, cha_extract_file, obd_file, col_name, cal_start, cal_end, time_step=None):
        """extract a simulated streamflow from the output.rch file,
            store it in each channel file.

        Args:
            - rch_file (`str`): the path and name of the existing output file
            - channels (`list`): channel number in a list, e.g. [9, 60]
            - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
            - end_day ('str'): simulation end day e.g. '12/31/2000'
            - time_step (`str`): day, month, year

        Example:
            pest_utils.extract_month_stf('path', [9, 60], '1/1/1993', '12/31/2000')
        """ 
        if time_step is None:
            time_step = 'day'
            stfobd_file = 'stf_day.obd.csv'
        if time_step == 'month':
            stfobd_file = 'stf_mon.obd.csv'
        stf_obd = self.read_cha_obd(obd_file)
        # stf_obd = get_last_day_of_month(stf_obd)
        stf_obd = stf_obd[cal_start:cal_end]
        stf_sim = pd.read_csv(
                            cha_extract_file,
                            sep=r'\s+',
                            names=["date", "stf_sim"],
                            index_col=0,
                            parse_dates=True)
        # result = pd.concat([stf_obd, stf_sim], axis=1)
        result = pd.concat([stf_sim, stf_obd], axis=1)
        result['tdate'] = pd.to_datetime(result.index)
        result['month'] = result['tdate'].dt.month
        result['year'] = result['tdate'].dt.year
        result['day'] = result['tdate'].dt.day
        if time_step == 'day':
            result['ins'] = (
                            'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                            result["month"].map('{:02d}'.format) +
                            result["day"].map('{:02d}'.format) + '!'
                            )
        elif time_step == 'month':
            result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
        else:
            print('are you performing a yearly calibration?')
        result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])
        with open(cha_extract_file+'.ins', "w", newline='') as f:
            f.write("pif ~" + "\n")
            result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
        print('{}.ins file has been created...'.format(cha_extract_file))
        return result['{}_ins'.format(col_name)]


    def irr_obd_to_ins(self, irr_extract_file, obd_file, col_name, cal_start, cal_end, time_step=None):
        """extract a simulated streamflow from the output.rch file,
            store it in each channel file.

        Args:
            - rch_file (`str`): the path and name of the existing output file
            - channels (`list`): channel number in a list, e.g. [9, 60]
            - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
            - end_day ('str'): simulation end day e.g. '12/31/2000'
            - time_step (`str`): day, month, year

        Example:
            pest_utils.extract_month_stf('path', [9, 60], '1/1/1993', '12/31/2000')
        """ 
        if time_step is None:
            time_step = 'day'
            stfobd_file = 'irr_paddy_day.obd.csv'
        if time_step == 'month':
            stfobd_file = 'irr_paddy_mon.obd.csv'
        stf_obd = self.read_cha_obd(obd_file)
        stf_obd = get_last_day_of_month(stf_obd)
        stf_obd = stf_obd[cal_start:cal_end]
        stf_sim = pd.read_csv(
                            irr_extract_file,
                            sep=r'\s+',
                            names=["date", "irr_sim"],
                            index_col=0,
                            parse_dates=True)
        # result = pd.concat([stf_obd, stf_sim], axis=1)
        result = pd.concat([stf_sim, stf_obd], axis=1)
        result['tdate'] = pd.to_datetime(result.index)
        result['month'] = result['tdate'].dt.month
        result['year'] = result['tdate'].dt.year
        result['day'] = result['tdate'].dt.day
        if time_step == 'day':
            result['ins'] = (
                            'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                            result["month"].map('{:02d}'.format) +
                            result["day"].map('{:02d}'.format) + '!'
                            )
        elif time_step == 'month':
            result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
        else:
            print('are you performing a yearly calibration?')
        result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])
        with open(irr_extract_file+'.ins', "w", newline='') as f:
            f.write("pif ~" + "\n")
            result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
        print('{}.ins file has been created...'.format(irr_extract_file))
        return result['{}_ins'.format(col_name)]


    def mf_obd_to_ins(
            self, wt_file, col_name, cal_start, cal_end,
            time_step="day", gType="waterlevel"
            ):
        """extract a simulated groundwater levels from the  file,
            store it in each channel file.

        Args:
            - rch_file (`str`): the path and name of the existing output file
            - channels (`list`): channel number in a list, e.g. [9, 60]
            - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
            - end_day ('str'): simulation end day e.g. '12/31/2000'

        Example:
            pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
        """ 
        if gType != "waterlevel":
            print("gType should be waterlevel")
            gwlType = "dtw"
            if time_step == "day":
                mf_obd_file = f"{gwlType}_day.obd.csv"
            if time_step == "month":
                mf_obd_file = f"{gwlType}_mon.obd.csv"
        else:
            print("gType is waterlevel")
            gwlType = "gwl"
            if time_step == "day":
                mf_obd_file = f"{gwlType}_day.obd.csv"
            if time_step == "month":
                mf_obd_file = f"{gwlType}_mon.obd.csv"   


        print(gwlType, mf_obd_file)         

        mf_obd = pd.read_csv(
                            mf_obd_file,
                            usecols=['date', col_name],
                            index_col=0,
                            na_values=[-999, ""],
                            parse_dates=True,
                            )
        mf_obd = mf_obd[cal_start:cal_end]

        wt_sim = pd.read_csv(
                            wt_file,
                            delim_whitespace=True,
                            names=["date", "stf_sim"],
                            index_col=0,
                            parse_dates=True)

        result = pd.concat([mf_obd, wt_sim], axis=1)

        result['tdate'] = pd.to_datetime(result.index)
        result['day'] = result['tdate'].dt.day
        result['month'] = result['tdate'].dt.month
        result['year'] = result['tdate'].dt.year
        result['ins'] = (
                        'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )
        result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

        with open(wt_file+'.ins', "w", newline='') as f:
            f.write("pif ~" + "\n")
            result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
        print('{}.ins file has been created...'.format(wt_file))

        return result['{}_ins'.format(col_name)]



    def read_cal(self):
        return pd.read_csv(
                        'calibration.cal',
                        sep=r'\s+',
                        skiprows=3,
                        header=None
                        )        

    def cal_to_tpl_file(self, tpl_file=None):
        """write a template file for a SWAT+ parameter value file (calibration.cal).

        Args:
            cal_file (`str`): the path and name of the existing model.in file
            tpl_file (`str`, optional):  template file to write. If None, use
                `cal_file` +".tpl". Default is None
        Note:
            Uses names in the first column in the pval file as par names.

        Example:
            pest_utils.model_in_to_template_file('path')

        Returns:
            **pandas.DataFrame**: a dataFrame with template file information
        """
        cal_file = 'calibration.cal'

        with open(cal_file, 'r') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()

        if tpl_file is None:
            tpl_file = cal_file + ".tpl"
        cal_df = self.read_cal()
        cal_df.index = cal_df.iloc[:, 0]
        # cal_df.iloc[:, 2] = cal_df.iloc[:, 0].apply(lambda x: " ~   {0:15s}   ~".format(x))
        cal_df.iloc[:, 2] = cal_df.iloc[:, 0].apply(lambda x: " ~ {0:12s} ~".format(x))

        # # cal_df.loc[:, "tpl"] = cal_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x[3:7]))

        with open(tpl_file, 'w') as f:
            f.write("ptf ~\n")
            f.write(line1)
            f.write(line2)
            f.write(line3)
            # f.write("{0:10d} #NP\n".format(cal_df.shape[0]))
            SFMT_LONG = lambda x: "{0:<10s} ".format(str(x))
            f.write(cal_df.loc[:, :].to_string(
                                    col_space=0,
                                    formatters=[SFMT_LONG]*len(cal_df.columns),
                                    index=False,
                                    header=False,
                                    justify="left"))
        # '''

        return cal_df


    def read_cal_parms(self):
        return pd.read_csv(
                        'cal_parms.cal',
                        sep=r'\s+',
                        skiprows=2,
                        )     

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
            wf.write("ptf ~\n")
            wf.writelines(data)
        print('  Finished ...')

    def replace_line(self, parnam, line):
        parts = line.split()
        new_line = (
            f'{int(parts[0]):<4d}' + f' ~ {parnam}{int(parts[0]):03d} ~ ' + "\n"
            )
        return new_line



    # NOTE: I am working on this function
    def update_par_inits_rgs(self, adjust_par):
        
        cal_adj = self.read_cal()
        cal_adj.rename(columns={0: "cal_parm", 1: "chg_type", 2: "cal_val"}, inplace=True)
        cal_db = self.read_cal_parms()
        for i in cal_adj.index:
            par = cal_adj.loc[i, "cal_parm"]
            abs_min = cal_db.loc[cal_db["name"]==par, "abs_min"]
            print(abs_min)
            
        

        # df_par = self.read_cal()





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
    # wd = "/Users/seonggyu.park/Documents/projects/tools/swatp-pest_wf/models/TxtInOut_Imsil_rye_rot_r1"
    wd = "/Users/seonggyu.park/Documents/projects/jj_test/main_opt"
    cns =  [1]
    cali_start_day = "1/1/2013"
    cali_end_day = "12/31/2023"
    obd_file = "singi_obs_q1_colnam.csv"
    obd_colnam = "cha01"
    cha_ext_file = "stf_001.txt"

    m1 = PstUtil(wd)
    io_files = pyemu.helpers.parse_dir_for_io_files('.')
    pst = pyemu.Pst.from_io_files(*io_files)
    par = pst.parameter_data
    m1.update_par_inits_rgs(par)

    print(par)

