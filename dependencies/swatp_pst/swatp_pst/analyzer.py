import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pyemu
import os
import matplotlib.dates as mdates

# uncertainty
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




# data comes from hanlder module and SWATMFout class
    
def create_stf_opt_df(pst, pt_oe, opt_idx=None):
    if opt_idx is None:
        opt_idx = -1
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
    pt_ut = pt_oe.iloc[opt_idx].T
    opt_df = pd.DataFrame()
    opt_df = pd.concat([pt_ut, obs], axis=1)
    return opt_df


def plot_observed_data(ax, df3, obd_col='obsval'):
    size = 10
    ax.plot(
        df3.time.values, df3[obd_col].values, c='m', lw=1.5, alpha=0.5,
        label="Observed", zorder=3
    )
    # ax.scatter(
    #     df3.index.values, df3[obd_col].values, c='m', lw=1, alpha=0.5, s=size, marker='x',
    #     label="Observed", zorder=3
    # )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    if len(df3[obd_col]) > 1:
        calculate_metrics(ax, df3, obd_col)
    else:
        display_no_data_message(ax)

def plot_stf_sim_obd(ax, stf_obd_df, obd_col):
    ax.plot(stf_obd_df.time.values, stf_obd_df.base.values, c='limegreen', lw=1, label="Simulated")
    plot_observed_data(ax, stf_obd_df, obd_col)
    # except Exception as e:
    #     handle_exception(ax, str(e))

def plot_stf_sim(ax, stf_df):
    try:
        ax.plot(stf_df.index.values, stf_df.base.values, c='limegreen', lw=1, label="Simulated")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    except Exception as e:
        handle_exception(ax, str(e)) 





# NOTE: metrics =======================================================================================
def calculate_metrics(ax, df3, obd_col):
    r_squared = ((sum((df3[obd_col] - df3[obd_col].mean()) * (df3.base - df3.base.mean())))**2) / (
            (sum((df3[obd_col] - df3[obd_col].mean())**2) * (sum((df3.base - df3.base.mean())**2)))
    )
    dNS = 1 - (sum((df3.base - df3[obd_col])**2) / sum((df3[obd_col] - (df3[obd_col]).mean())**2))
    PBIAS = 100 * (sum(df3[obd_col] - df3.base) / sum(df3[obd_col]))
    display_metrics(ax, dNS, r_squared, PBIAS)

def calculate_metrics_gw(ax, df3, grid_id, obd_col):
    r_squared = ((sum((df3[obd_col] - df3[obd_col].mean()) * (df3[str(grid_id)] - df3[str(grid_id)].mean())))**2) / (
            (sum((df3[obd_col] - df3[obd_col].mean())**2) * (sum((df3[str(grid_id)] - df3[str(grid_id)].mean())**2)))
    )
    dNS = 1 - (sum((df3[str(grid_id)] - df3[obd_col])**2) / sum((df3[obd_col] - (df3[obd_col]).mean())**2))
    PBIAS = 100 * (sum(df3[obd_col] - df3[str(grid_id)]) / sum(df3[obd_col]))
    display_metrics(ax, dNS, r_squared, PBIAS)

def display_metrics(ax, dNS, r_squared, PBIAS):
    ax.text(
        .01, 0.95, f'Nash-Sutcliffe: {dNS:.4f}',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .01, 0.90, f'$R^2$: {r_squared:.4f}',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .99, 0.95, f'PBIAS: {PBIAS:.4f}',
        fontsize=8, horizontalalignment='right', color='limegreen', transform=ax.transAxes
    )

def display_no_data_message(ax):
    ax.text(
        .01, .95, 'Nash-Sutcliffe: ---',
        fontsize=8, horizontalalignment='left', transform=ax.transAxes
    )
    ax.text(
        .01, 0.90, '$R^2$: ---',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .99, 0.95, 'PBIAS: ---',
        fontsize=8, horizontalalignment='right', color='limegreen', transform=ax.transAxes
    )

def handle_exception(ax, exception_message):
    ax.text(
        .5, .5, exception_message,
        fontsize=12, horizontalalignment='center', weight='extra bold', color='y', transform=ax.transAxes
    )

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)



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
        pst=pst,filename=os.path.join(m_d,"swatp_nw_ies.{0}.obs.csv".format(4)))
    

    df = create_stf_opt_df(pst, pt_oe)
    obd_col = "obsval"
    fig, ax = plt.subplots()
    plot_stf_sim_obd(ax, df, obd_col)
    plt.show()
    # single_plot_tseries_ensembles(pst, pr_oe, pt_oe, width=10, height=4, dot=False)