import matplotlib.pyplot as plt
import math
import numpy as np
import pyemu
import os


def single_plot_tseries_ensembles(
                    pst, pr_oe, pt_oe, width=10, height=4, dot=True,
#                     onames=["hds","sfr"]
                    ):
    # pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
    print(obs)

#     # onames provided in oname argument
#     obs = obs.loc[obs.oname.apply(lambda x: x in onames)]
    # only non-zero observations
#     obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    # make a plot
    ogs = obs.obgnme.unique()
    fig, ax = plt.subplots(figsize=(width,height))

    oobs = obs
    oobs.loc[:,"time"] = oobs.loc[:,"time"].astype(str)
#         oobs.sort_values(by="time",inplace=True)
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    # '''
    if dot is True:
        # plot prior
        [ax.scatter(tvals,pr_oe.loc[i,onames].values,color="gray",s=30, alpha=0.5) for i in pr_oe.index]
        # plot posterior
        [ax.scatter(tvals,pt_oe.loc[i,onames].values,color='b',s=30,alpha=0.2) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        ax.scatter(oobs.time,oobs.obsval,color='red',s=30).set_facecolor("none")
    if dot is False:
        # plot prior
        [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in pr_oe.index]
        # plot posterior
        [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.5,alpha=0.5) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
    ax.tick_params(axis='x', labelrotation=90)
    ax.margins(x=0.01)
    # ax.set_title(og,loc="left")
    # fig.tight_layout()
    plt.show()
    # '''

def plot_tseries_ensembles(
                    pst, pr_oe, pt_oe, width=10, height=4, dot=True,
#                     onames=["hds","sfr"]
                    ):
    # pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
#     # onames provided in oname argument
#     obs = obs.loc[obs.oname.apply(lambda x: x in onames)]
    # only non-zero observations
#     obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    # make a plot
    ogs = obs.obgnme.unique()
    fig,axes = plt.subplots(len(ogs),1,figsize=(width,height*len(ogs)), squeeze=False)
    ogs.sort()
    # for each observation group (i.e. timeseries)
    for ax,og in zip(axes,ogs):
        # get values for x axis
        oobs = obs.loc[obs.obgnme==og,:].copy()
        oobs.loc[:,"time"] = oobs.loc[:,"time"].astype(str)
#         oobs.sort_values(by="time",inplace=True)
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        if dot is True:
            # plot prior
            [ax.scatter(tvals,pr_oe.loc[i,onames].values,color="gray",s=30, alpha=0.5) for i in pr_oe.index]
            # plot posterior
            [ax.scatter(tvals,pt_oe.loc[i,onames].values,color='b',s=30,alpha=0.2) for i in pt_oe.index]
            # plot measured+noise 
            oobs = oobs.loc[oobs.weight>0,:]
            tvals = oobs.time.values
            onames = oobs.obsnme.values
            ax.scatter(oobs.time,oobs.obsval,color='red',s=30).set_facecolor("none")
        if dot is False:
            # plot prior
            [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in pr_oe.index]
            # plot posterior
            [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.5,alpha=0.5) for i in pt_oe.index]
            # plot measured+noise 
            oobs = oobs.loc[oobs.weight>0,:]
            tvals = oobs.time.values
            onames = oobs.obsnme.values
            ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
        ax.tick_params(axis='x', labelrotation=90)
        ax.margins(x=0.01)
        ax.set_title(og,loc="left")
    # fig.tight_layout()
    plt.show()

def plot_prior_posterior_par_hist(prior_df, post_df, sel_pars, width=7, height=5, ncols=3):
    nrows = math.ceil(len(sel_pars)/ncols)
    fig, axes = plt.subplots(figsize=(width, height), nrows=nrows, ncols=ncols)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, ax in enumerate(axes.flat):
        if i<len(sel_pars):
            colnam = sel_pars['parnme'].tolist()[i]
            ax.hist(prior_df.loc[:, colnam].values,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0], 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0], 20),
                    color = "gray", alpha=0.5, density=True,
                    label="Prior"
            )
            y, x, _ = ax.hist(post_df.loc[:, colnam].values,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0], 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0], 20), 
                     alpha=0.5, density=True, label="Posterior"
            )
            ax.set_ylabel(colnam)
            ax.set_yticks([])
    plt.xlabel("Parameter range")
    plt.show()



if __name__ == '__main__':
    # wd = "/Users/seonggyu.park/Documents/projects/tools/swatp-pest_wf/models/TxtInOut_Imsil_rye_rot_r1"
    wd = "/Users/seonggyu.park/Documents/projects/jj_test/main_opt"
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
    # m1.update_par_initials_ranges(par)

    # print(par)
    m_d = '/Users/seonggyu.park/Documents/projects/jj/swatp_nw_ies'
    pst_file = "swatp_nw_ies.pst"
    pst = pyemu.Pst(os.path.join(m_d, pst_file))
    # load prior simulation
    pr_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(m_d,"swatp_nw_ies.0.obs.csv")
        )
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(m_d,"swatp_nw_ies.{0}.obs.csv".format(2)))
    single_plot_tseries_ensembles(pst, pr_oe, pt_oe, width=10, height=4, dot=False)