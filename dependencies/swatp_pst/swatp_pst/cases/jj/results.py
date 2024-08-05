import pyemu
import os


def result_ies_tot():
    # info
    wd = 'D:\\jj\\Albufera\\alb_nw_ies'
    pst_file = "alb_nw_ies.pst"
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    # load prior simulation
    pr_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"alb_nw_ies.0.obs.csv")
        )
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"alb_nw_ies.{0}.obs.csv".format(2)))
    

    # --------
    # RESULT01
    #---------
    iter_idx = 1
    par_obj_file = f"alb_nw_ies.{iter_idx}.par.objs.csv"
    bstcs=["46"]
    parbds = [20, 180]
    nsbds = [0.70, 1]
    pbiasbds = [-15, 15]
    # # check best realization
    # fter = filter_candidates(
    #     wd, pst, par_obj_file, parbds=parbds,
    #     nsbds=None, pbiasbds=None,
    #     rsqbds=None, rmsebds=None,
    #     savefile=False)
    # for rel_idx in fter.loc[:, "real_name"]:
    #     print(f"cal:{rel_idx}",
    #         get_rels_cal_val_objs(
    #         wd, pst_file, 
    #         iter_idx=iter_idx, 
    #         opt_idx=rel_idx, 
    #         calval="cal")
    #         )
    #     print(f"val:{rel_idx}",
    #         get_rels_cal_val_objs(
    #         wd, pst_file, 
    #         iter_idx=iter_idx, 
    #         opt_idx=rel_idx, 
    #         calval="val")
    #         )

    df = get_pr_pt_df(pst, pr_oe, pt_oe)
    pcp_df = create_pcp_df(wd, 1)
    plot_fill_between_ensembles(
        df, 
        # pcp_df=pcp_df,
        # caldates=['1/1/2017','12/31/2023'],
        # valdates=['1/1/2013','12/31/2016'],
        size=20
        )
    # get_p_factor(pst, pt_oe, perc_obd_nz=None, cal_val=True)
    # get_d_factor(pst, pt_oe, cal_val=True)

    # objs = ['ns', 'pbias', 'rsq', 'rmse']
    # # plot par obj
    # for obj in objs:
    #     plot_par_obj(wd, pst,  par_obj_file, 
    #         objf=obj, width=7, height=8, ncols=3,
    #         bstcs=bstcs, orgsim=None,
    #         save_fig=True)


    # iter_idx = 5
    # # create_rels_objs(wd, pst_file, iter_idx)
    # filter_candidates(
    #     wd, pst, par_obj_file, parbds=parbds,
    #     nsbds=None, pbiasbds=None,
    #     rsqbds=None, rmsebds=None,
    #     savefile=True)

def plot_prior_posterior_par_hist_jj(
        width=4, height=3, ncols=3, bestcand=None, parobj_file=None, sharey=True):
    
    wd1 = 'D:\\jj\\opt_3rd\\optimized_results'
    pstfile1 = "swatp_nw_ies.pst"
    iter_idx1 = 5
    pst1 = pyemu.Pst(os.path.join(wd1, pstfile1))
    pst_nam1= pstfile1[:-4]
    prior_df1 = pyemu.ParameterEnsemble.from_csv(
        pst=pst1,filename=os.path.join(wd1,f"{pst_nam1}.{0}.par.csv"))
    post_df1 = pyemu.ParameterEnsemble.from_csv(
        pst=pst1,filename=os.path.join(wd1,f"{pst_nam1}.{iter_idx1}.par.csv"))
    df_pars1 = pd.read_csv(os.path.join(wd1, f"{pst_nam1}.par_data.csv"))
    sel_pars1 = df_pars1.loc[df_pars1["partrans"]=='log']
    bstcs1="glm"
    par_obj_file1 = f"{pst_nam1}.{iter_idx1}.par.objs.csv"
    wd2 = 'D:\\jj\\Albufera\\alb_nw_calibrated'
    pstfile2 = "alb_nw_ies.pst"
    iter_idx2 = 4
    pst2 = pyemu.Pst(os.path.join(wd2, pstfile2))

    pst_nam2= pstfile2[:-4]
    prior_df2 = pyemu.ParameterEnsemble.from_csv(
        pst=pst2,filename=os.path.join(wd2,f"{pst_nam2}.{0}.par.csv"))
    post_df2 = pyemu.ParameterEnsemble.from_csv(
        pst=pst2,filename=os.path.join(wd2,f"{pst_nam2}.{3}.par.csv"))
    df_pars2 = pd.read_csv(os.path.join(wd2, f"{pst_nam2}.par_data.csv"))
    sel_pars2 = df_pars2.loc[df_pars2["partrans"]=='log']
    bstcs2="glm"
    par_obj_file2 = f"{pst_nam2}.{iter_idx2}.par.objs.csv"

    pars_info1 = get_par_offset(pst1)
    pars_info2 = get_par_offset(pst2)
    colnams = [
        "alpha", "awc", "cn2", "cn3_swf", "epco", 
        "esco", "lat_len", "latq_co", "perco", "petco"
    ]
    for colnam in colnams:
        fig, axes = plt.subplots(figsize=(width, height), nrows=1, ncols=2, sharey=True)
        ax1 = fig.add_subplot(111, frameon=False)
        ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        offset = pars_info1.loc[colnam, "offset"]
        axes[0].hist(prior_df1.loc[:, colnam].values + offset,
                bins=np.linspace(
                    sel_pars1.loc[sel_pars1["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                    sel_pars1.loc[sel_pars1["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20),
                color = "gray", alpha=0.5, density=False,
                label="Prior", orientation='horizontal'
        )
        y, x, _ = axes[0].hist(post_df1.loc[:, colnam].values + offset,
                bins=np.linspace(
                    sel_pars1.loc[sel_pars1["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                    sel_pars1.loc[sel_pars1["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20), 
                    alpha=0.5, density=False, label="Posterior", orientation='horizontal'
        )
        po_df1 = pd.read_csv(os.path.join(wd1, par_obj_file1))
        x1 = po_df1.loc[po_df1["real_name"]=="46", colnam].values + offset
        axes[0].axhline(y=x1, color='r', linestyle="--", alpha=0.5)
        axes[0].invert_xaxis()
        # axes[0].yaxis.set_label_position("right")
        # axes[0].yaxis.tick_right()
        # for tick in axes[0].yaxis.get_majorticklabels():
        #     tick.set_horizontalalignment("left")

        axes[0].set_title(colnam, fontsize=12, ha='left', x=0.07, y=0.97, backgroundcolor='white')

        offset = pars_info2.loc[colnam, "offset"]
        axes[1].hist(prior_df2.loc[:, colnam].values + offset,
                bins=np.linspace(
                    sel_pars2.loc[sel_pars2["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                    sel_pars2.loc[sel_pars2["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20),
                color = "gray", alpha=0.5, density=False,
                label="Prior", orientation='horizontal'
        )
        y, x, _ = axes[1].hist(post_df2.loc[:, colnam].values + offset,
                bins=np.linspace(
                    sel_pars2.loc[sel_pars2["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                    sel_pars2.loc[sel_pars2["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20), 
                    alpha=0.5, density=False, label="Posterior", orientation='horizontal',
                    color = "C1"
        )
        po_df2 = pd.read_csv(os.path.join(wd2, par_obj_file2))
        x2 = po_df2.loc[po_df2["real_name"]=="glm", colnam].values + offset
        axes[1].axhline(y=x2, color='r', linestyle="--", alpha=0.5)
        for ax in axes:
            ax.tick_params(axis='both', labelsize=12)
            ax.yaxis.set_major_locator(FixedLocator([-100, 0, 100]))
        # ax.tick_params(axis='x', labelsize=8)       
        # plt.xlabel(r"Frequency", fontsize=10)
        # plt.ylabel(r"Parameter relative change (%)", fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        plt.savefig(f'{colnam}_hist.png', bbox_inches='tight', dpi=300)
        plt.show()
        print(os.getcwd())





def albufera_par_results():
    wd = 'D:\\jj\\Albufera\\alb_nw_calibrated'
    pstfile = "alb_nw_ies.pst"
    iter_idx = 4
    # get_average_annual_wb(wd, colnam)
  
    pst = pyemu.Pst(os.path.join(wd, pstfile))
    pst_nam = pstfile[:-4]
    prior_df = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,f"{pst_nam}.{0}.par.csv"))
    post_df = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,f"{pst_nam}.{3}.par.csv"))
    df_pars = pd.read_csv(os.path.join(wd, f"{pst_nam}.par_data.csv"))
    sel_pars = df_pars.loc[df_pars["partrans"]=='log']
    bstcs="glm"
    par_obj_file = f"{pst_nam}.{iter_idx}.par.objs.csv"
    plot_prior_posterior_par_hist(wd,
                                pst, prior_df, post_df, sel_pars,
                                width=9, height=5, ncols=5,
                                bestcand=bstcs, parobj_file=par_obj_file)





if __name__ == '__main__':
    # info
    # wd = '/Users/seonggyu.park/Documents/projects/jj/swatp_nw_sen_sobol_1500'
    # wd = 'D:\\jj\\opt_3rd\\swatp_nw_ies'
    # pst_file = "swatp_nw_ies.pst"
    # pst = pyemu.Pst(os.path.join(wd, pst_file))

    # m_d2 = 'D:\\jj\\TxtInOut_Imsil_rye_rot_r2'
    # org_sim = create_stf_sim_obd_df(m_d2, 1, "singi_obs_q1_colnam.csv", "cha01")
    # cal_sim = create_stf_sim_obd_df(wd, 1, "singi_obs_q1_colnam.csv", "cha01")
    # print(cal_sim)
    # # plot progress comparison pre and post
    # fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # plot_stf_sim_obd(axes[0], org_sim, dot=True)
    # plot_stf_sim_obd(axes[1], cal_sim, dot=True)
    # axes[0].set_title('pre')
    # axes[1].set_title('post')
    # plt.show()
    # plot_sen_sobol(wd, pst_file)

    # wd = '/Users/seonggyu.park/Documents/projects/tools/swatp_pst_wf/models/calibrated_model'
    # obd_file = "singi_obs_q1_colnam.csv"
    # obd_colnam = "cha01"
    # cha_id = 1

    # df = create_stf_sim_obd_df(wd, cha_id, obd_file, obd_colnam)
    # # print(df)
    # validates = ['1/1/2013', '12/31/2016']
    # calidates = ['1/1/2017', '12/31/2023']

    # colnam = "surq_cha"

    # fig, ax = plt.subplots()
    # # plot_wb_mon_cal_val_hist(ax, wd, colnam, calidates, validates)
    # plot_flow_cal_val_hist(ax, df, calidates, validates)
    # plt.show()
    
    

    plot_prior_posterior_par_hist_jj()
    
    
    
    wd = 'D:\\jj\\opt_3rd\\swatp_nw_sen_sobol'
    pstfile = "swatp_nw_sen_sobol.pst"
    wd2 = 'D:\\jj\\Albufera\\alb_nw_sen_sobol'
    pstfile2 = "alb_nw_sen_sobol.pst"
    # sftsm, sttsm, sfts_cfis, stts_cfis = get_sobol_results(wd, pst_file)
    # plot_sen_sobol_jj(wd, pstfile, wd2, pstfile2)

    
    # result_ies()
    # wd = 'D:\\jj\\Albufera\\alb_nw_ies'
    # pst_file = "alb_nw_ies.pst"
    # create_rels_objs(wd, pst_file, 4)
    # result_ies_tot()
    # result par
    '''
    prior_df = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"swatp_nw_ies.{0}.par.csv".format(0)))
    post_df = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"swatp_nw_ies.{0}.par.csv".format(5)))
    df_pars = pd.read_csv(os.path.join(wd, "swatp_nw_ies.par_data.csv"))
    sel_pars = df_pars.loc[df_pars["partrans"]=='log']
    plot_prior_posterior_par_hist(
                                pst, prior_df, post_df, sel_pars,
                                width=9, height=5, ncols=5,
                                bestcand=bstcs, parobj_file=par_obj_file)
    '''

    # result_ies_tot()