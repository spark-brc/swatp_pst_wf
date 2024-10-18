import pandas as pd
from swatp_pst import handler
import calendar
import numpy as np
import os
import pyemu


class PstUtil:
    def __init__(self, wd):
        self.wd = wd
        os.chdir(self.wd)


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

    def crop_aa_obd_to_ins(self, crop_aa_extract_file):
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

        stf_sim = pd.read_csv(
                            crop_aa_extract_file,
                            sep=r'\s+',
                            names=["hruid", "crop", "mass", "c", 'n', 'p']
                            )
        stf_sim['ins'] = (
                        'l1 w w  !' + stf_sim["hruid"].map('{:05d}'.format) +
                        stf_sim["crop"].map(str) + '! w w w'
                        )
        with open(crop_aa_extract_file+'.ins', "w", newline='') as f:
            f.write("pif ~" + "\n")
            stf_sim['ins'].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
        print('{}.ins file has been created...'.format(crop_aa_extract_file))
        return stf_sim['ins']






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

    def update_par_inits_rgs(self, precal_df):
        
        cal_adj = self.read_cal()
        cal_db = self.read_cal_parms()
        for i in cal_adj.index:
            par = cal_adj.loc[i, "cal_parm"]
            abs_min = cal_db.loc[cal_db["name"]==par, "abs_min"]
            print(abs_min)

    def read_cal(self):
        return pd.read_csv(
                        'calibration.cal',
                        sep=r'\s+',
                        skiprows=3,
                        header=None
                        )       
        # df_par = self.read_cal()

    def plant_to_tpl_file(self, crop):
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
        plt_file = 'plants.plt'
        tpl_file = plt_file + ".tpl"
        with open(plt_file, "r") as inf:
            data = inf.readlines()
            c = 0
            colnams = data[1].split()
            for line in data:
                if line.split()[0] == crop:
                    new_line = self.replace_line_plt(line, colnams)
                    data[c] = new_line
                c += 1
        with open(tpl_file, "w") as wf:
            wf.write('ptf ~\n')
            wf.writelines(data)
        new_file = os.path.join(self.wd, tpl_file)
        print(
            f" {'>'*3} {os.path.basename(new_file)}" + 
            " file is created ..."
            )

    def replace_line_plt(self, line, colnams):
        parts = line.split()
        newline = []
        for i in range(len(parts)):
            if i == 0:
                newline.append(f'{parts[i]:<16s}')
            elif i == 1 or i == 2:
                newline.append(f'{parts[i]:>18s}')
            elif (i > 2) and i < (len(parts) -1):
                newline.append(f" ~{colnams[i]:>11s}~" )
            else:
                newline.append(f"  {'rice':<14s}" )
        newline.append("\n")
        newline ="".join(newline)            
        return newline

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
    # NOTE: for jj work
    # wd = "/Users/seonggyu.park/Documents/projects/jj_test/main_opt"
    # cns =  [1]
    # cali_start_day = "1/1/2013"
    # cali_end_day = "12/31/2023"
    # obd_file = "singi_obs_q1_colnam.csv"
    # obd_colnam = "cha01"
    # cha_ext_file = "stf_001.txt"

    # m1 = PstUtil(wd)
    # io_files = pyemu.helpers.parse_dir_for_io_files('.')
    # pst = pyemu.Pst.from_io_files(*io_files)
    # par = pst.parameter_data
    # m1.update_par_inits_rgs(par)
    # print(par)

    #NOTE: create template file for plant
    wd = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    m1 = PstUtil(wd)
    m1.plant_to_tpl_file('rice_dwn')
    m1.crop_aa_obd_to_ins("crop_aa_rice_dwn.txt")
