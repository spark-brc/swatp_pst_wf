import os
import pandas as pd

def create_swatp_pst_con(prj_dir, swatp_wd, cal_start, cal_end, chs, time_step='month'):
    col01 = [
        'prj_dir',
        'swatp_wd', 
        'cal_start', 
        'cal_end', 
        'chs',
        'time_step'
    ]
    col02 = [
        prj_dir,
        swatp_wd, 
        cal_start, 
        cal_end, 
        chs,
        time_step
    ]
    df = pd.DataFrame({'names': col01, 'vals': col02})

    main_opt_path = os.path.join(prj_dir, 'main_opt')
    if not os.path.isdir(main_opt_path):
        os.makedirs(main_opt_path)
    con_path = os.path.join(main_opt_path, 'swatp_pst.con')
    with open(con_path, 'w', newline='') as f:
        f.write("# swatp_pst.con created by swatp_pst\n")
        df.to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print(f"'swatp_pst.con' created at: {con_path}")
    return con_path

if __name__ == '__main__':
    prj_dir = r"C:\TxtInOut_Albuferajune24"
    swatp_wd = r"C:\swatmf\dependencies\swatp_pst\swatp_pst"
    cal_start = "1/1/2016"
    cal_end = "12/31/2022"
    chs = "[1]"
    time_step = "month"

    create_swatp_pst_con(prj_dir, swatp_wd, cal_start, cal_end, chs, time_step)
