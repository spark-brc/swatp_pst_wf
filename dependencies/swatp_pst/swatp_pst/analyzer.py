import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pyemu
import os
import matplotlib.dates as mdates
from swatp_pst.handler import SWATp
from swatp_pst import objfns

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

def single_plot_tseries_ensembles_plots_added(
                    pst, pr_oe, pt_oe, 
                    width=10, height=4, dot=True,
                    size=None, bstcs=None,
                    orgsim=None
                    ):
    if size is None:
        size = 30
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
        # ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
        ax.scatter(oobs.time,oobs.obsval,color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
        if bstcs is not None:
            [ax.plot(tvals,pt_oe.loc[i,onames].values, lw=1, label=i) for i in bstcs]
        if orgsim is not None:
            orgsim = orgsim
            ax.plot(tvals, orgsim.iloc[:, 0].values, c='limegreen', lw=1, label="Original")

    
    ax.tick_params(axis='x', labelrotation=90)
    ax.margins(x=0.01)
    plt.legend()
    # ax.set_title(og,loc="left")
    # fig.tight_layout()
    plt.show()


def single_plot_fdc_added(
                    pst, pr_oe, pt_oe, 
                    width=10, height=8, dot=True,
                    size=None, bstcs=None,
                    orgsim=None
                    ):
    if size is None:
        size = 30
    # pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
    onames = obs.obsnme.values

    obs_d, obd_exd = convert_fdc_data(obs.obsval.values)
    pr_min_d, pr_min_exd = convert_fdc_data(pr_oe.min().values)
    pr_max_d, pr_max_exd = convert_fdc_data(pr_oe.max().values)
    pt_min_d, pt_min_exd = convert_fdc_data(pt_oe.min().values)
    pt_max_d, pt_max_exd = convert_fdc_data(pt_oe.max().values)

    fig, ax = plt.subplots(figsize=(width,height))
    ax.fill_between(pr_min_exd*100, pr_min_d, pr_max_d, interpolate=False, facecolor="0.5", alpha=0.4)
    ax.fill_between(pt_min_exd*100, pt_min_d, pt_max_d, interpolate=False, facecolor="b", alpha=0.4)
    ax.scatter(obd_exd*100, obs_d, color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
    if orgsim is not None:
        orgsim = orgsim
        org_d, org_exd = convert_fdc_data(orgsim.iloc[:, 0].values)
        ax.plot(org_exd*100, org_d, c='limegreen', lw=2, label="Original")
    if bstcs is not None:
        for bstc in bstcs:
            dd, eexd = convert_fdc_data(pt_oe.loc[bstc,onames].values)
            ax.plot(eexd*100, dd, lw=2, label=bstc)

    ax.set_yscale('log')
    ax.set_xlabel(r"Exceedence [%]", fontsize=12)
    ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
    ax.margins(0.01)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12, loc="lower left")
    plt.tight_layout()
    plt.savefig('fdc.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(os.getcwd())

    # return pr_oe_min


def convert_fdc_data(data):
    data = np.sort(data)[::-1]
    exd = np.arange(1.,len(data)+1) / len(data)
    return data, exd




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

def plot_prior_posterior_par_hist(pst, prior_df, post_df, sel_pars, width=7, height=5, ncols=3):
    nrows = math.ceil(len(sel_pars)/ncols)
    pars_info = get_par_offset(pst)
    fig, axes = plt.subplots(figsize=(width, height), nrows=nrows, ncols=ncols)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, ax in enumerate(axes.flat):
        if i<len(sel_pars):
            colnam = sel_pars['parnme'].tolist()[i]
            offset = pars_info.loc[colnam, "offset"]
            
            ax.hist(prior_df.loc[:, colnam].values + offset,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20),
                    color = "gray", alpha=0.5, density=True,
                    label="Prior"
            )
            y, x, _ = ax.hist(post_df.loc[:, colnam].values + offset,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20), 
                     alpha=0.5, density=True, label="Posterior"
            )
            ax.set_ylabel(colnam)
            ax.set_yticks([])
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
    plt.xlabel("Parameter range")
    plt.show()




# data comes from hanlder module and SWATMFout class
    
def create_stf_sim_obd_df(wd, cha_id, obd_file, obd_col):
    m1 = SWATp(wd)
    start_day = m1.stdate_warmup
    sim = m1.read_cha_morph_mon()
    sim = sim.loc[sim["gis_id"] == cha_id]
    sim = sim.drop(['gis_id'], axis=1)
    sim.index = pd.date_range(start_day, periods=len(sim.flo_out), freq='ME')
    obd = m1.read_cha_obd(obd_file)
    obd = obd.loc[:, obd_col]
    opt_df = pd.DataFrame()
    opt_df = pd.concat([sim, obd], axis=1)
    opt_df["time"] = opt_df.index
    opt_df = opt_df.rename({obd_col: 'obsval'}, axis=1)
    opt_df.dropna(inplace=True)
    return opt_df

    
def create_stf_opt_df(pst, pt_oe, opt_idx=None):
    if opt_idx is None:
        opt_idx = -1
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
    pt_ut = pt_oe.loc[opt_idx].T
    opt_df = pd.DataFrame()
    opt_df = pd.concat([pt_ut, obs], axis=1)
    return opt_df


def get_rels_objs(wd, pst_file, iter_idx=None, opt_idx=None):
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    if iter_idx is None:
        iter_idx = pst.control_data.noptmax
    if opt_idx is None:
        opt_idx = -1
    
    # load observation data
    obs = pst.observation_data.copy()
    pst_nam = pst_file[:-4]
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))

    pt_ut = pt_oe.loc[opt_idx].T
    opt_df = pd.DataFrame()
    opt_df = pd.concat([pt_ut, obs], axis=1)
    sims = opt_df.iloc[:, 0].tolist()
    obds = opt_df.iloc[:, 2].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    return ns, pbias, rsq, rmse
    
def create_rels_objs(wd, pst_file, iter_idx):
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    # load observation data
    # obs = pst.observation_data.copy()
    pst_nam = pst_file[:-4]
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))
    pt_par = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.par.csv".format(pst_nam, iter_idx)))
    pt_oe_df = pd.DataFrame(pt_oe, index=pt_oe.index, columns=pt_oe.columns)
    pt_par_df = pd.DataFrame(pt_par, index=pt_oe.index, columns=pt_par.columns)
    nss = []
    pbiass = []
    rsqs = []
    rmses = []
    # for i in range(np.shape(pt_oe)[0]):
    for i in pt_oe.index:
        ns, pbias, rsq, rmse = get_rels_objs(wd, pst_file, iter_idx=iter_idx, opt_idx=i)
        nss.append(ns)
        pbiass.append(pbias)
        rsqs.append(rsq)
        rmses.append(rmse)
    objs_df = pd.DataFrame({"ns": nss, "pbias": pbiass, "rsq": rsqs, "rmse": rmses}, index=pt_oe.index)
    pt_oe_df = pd.concat([pt_oe_df, objs_df], axis=1)
    pt_par_df = pd.concat([pt_par_df, objs_df], axis=1)
    pt_oe_df.to_csv(os.path.join(wd, "{0}.{1}.obs.objs.csv".format(pst_nam, iter_idx)))
    pt_par_df.to_csv(os.path.join(wd, "{0}.{1}.par.objs.csv".format(pst_nam, iter_idx)))


def get_par_offset(pst):
    pars = pst.parameter_data.copy()
    pars = pars.loc[:, ["parnme", "offset"]]
    return pars
    
    



def plot_par_obj(wd, pst,  par_obj_file, objf=None, width=7, height=3, ncols=3):
    if objf is None:
        objf = "NS"
    po_df = pd.read_csv(os.path.join(wd, par_obj_file))
    pars_df = po_df.iloc[:, 1:-4]
    par_cols = pars_df.columns.values
    objs = po_df.loc[:, objf.lower()].values
    pars_info = get_par_offset(pst)
    nrows = math.ceil(len(par_cols)/ncols)
    fig, axes = plt.subplots(figsize=(width, height), nrows=nrows, ncols=ncols)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, ax in enumerate(axes.flat):
        if i<len(par_cols):
            offset = pars_info.iloc[i, 1]
            ax.scatter(pars_df.iloc[:, i] + offset ,objs,s=30,alpha=0.2)
            ax.set_title(par_cols[i])
            # ax.set_yticks([])
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)   
    plt.xlabel("Parameter relative change (%)")
    plt.tight_layout()
    plt.show()

def plot_observed_data(ax, df3, size=None, dot=False):
    if size is None:
        size = 10
    if dot is False:
        ax.plot(
            df3.loc[:, 'time'].values, df3.loc[:, "obsval"].values, c='m', lw=1.5, alpha=0.5,
            label="Observed", zorder=3
        )
    if dot is True:
        ax.scatter(
            df3.loc[:, 'time'].values, df3.loc[:, "obsval"].values, c='m', lw=1, alpha=0.5, s=size, marker='x',
            label="Observed", zorder=3
        )
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    if len(df3.loc[:, "obsval"]) > 1:
        calculate_metrics(ax, df3)
    else:
        display_no_data_message(ax)

def plot_stf_sim_obd(ax, opt_df, size=None, dot=False):
    if size is None:
        size = 10    
    ax.plot(opt_df.loc[:, 'time'].values, opt_df.iloc[:, 0].values, c='limegreen', lw=1, label="Simulated")
    if dot is False:
        dot = False
    if dot is True:
        dot = True
    # size=10
    plot_observed_data(ax, opt_df, size=size, dot=dot)
    # except Exception as e:
    #     handle_exception(ax, str(e))

def plot_stf_sim(ax, stf_df):
    try:
        ax.plot(stf_df.index.values, stf_df.base.values, c='limegreen', lw=1, label="Simulated")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    except Exception as e:
        handle_exception(ax, str(e)) 





# NOTE: metrics =======================================================================================
def calculate_metrics_opt(opt_df):
    r_squared = ((sum((opt_df.loc[:, "obsval"] - opt_df.loc[:, "obsval"].mean()) * (opt_df.iloc[:, 0] - opt_df.iloc[:, 0].mean())))**2) / (
            (sum((opt_df.loc[:, "obsval"] - opt_df.loc[:, "obsval"].mean())**2) * (sum((opt_df.iloc[:, 0] - opt_df.iloc[:, 0].mean())**2)))
    )
    dNS = 1 - (sum((opt_df.iloc[:, 0] - opt_df.loc[:, "obsval"])**2) / sum((opt_df.loc[:, "obsval"] - (opt_df.loc[:, "obsval"]).mean())**2))
    pbias = 100 * (sum(opt_df.loc[:, "obsval"] - opt_df.iloc[:, 0]) / sum(opt_df.loc[:, "obsval"]))





def calculate_metrics(ax, df3):
    r_squared = ((sum((df3.loc[:, "obsval"] - df3.loc[:, "obsval"].mean()) * (df3.iloc[:, 0] - df3.iloc[:, 0].mean())))**2) / (
            (sum((df3.loc[:, "obsval"] - df3.loc[:, "obsval"].mean())**2) * (sum((df3.iloc[:, 0] - df3.iloc[:, 0].mean())**2)))
    )
    dNS = 1 - (sum((df3.iloc[:, 0] - df3.loc[:, "obsval"])**2) / sum((df3.loc[:, "obsval"] - (df3.loc[:, "obsval"]).mean())**2))
    PBIAS = 100 * (sum(df3.loc[:, "obsval"] - df3.iloc[:, 0]) / sum(df3.loc[:, "obsval"]))
    display_metrics(ax, dNS, r_squared, PBIAS)

def calculate_metrics_gw(ax, df3, grid_id):
    r_squared = ((sum((df3.loc[:, "obsval"] - df3.loc[:, "obsval"].mean()) * (df3[str(grid_id)] - df3[str(grid_id)].mean())))**2) / (
            (sum((df3.loc[:, "obsval"] - df3.loc[:, "obsval"].mean())**2) * (sum((df3[str(grid_id)] - df3[str(grid_id)].mean())**2)))
    )
    dNS = 1 - (sum((df3[str(grid_id)] - df3.loc[:, "obsval"])**2) / sum((df3.loc[:, "obsval"] - (df3.loc[:, "obsval"]).mean())**2))
    PBIAS = 100 * (sum(df3.loc[:, "obsval"] - df3[str(grid_id)]) / sum(df3.loc[:, "obsval"]))
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
    # wd = "/Users/seonggyu.park/Documents/projects/jj_test/main_opt"
    # wd = "D:\\jj\\jj\\swatp_nw_ies"
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
    # cha_id =  1
    # obd_file = "singi_obs_q1_colnam.csv"
    # obd_col = "cha01"
    # df = create_stf_sim_obd_df(wd, cha_id, obd_file, obd_col)
    # print(df)    
    # # print(par)
    # m_d = '/Users/seonggyu.park/Documents/projects/jj/swatp_nw_ies'
    wd = 'D:\\jj\\opt_2nd\\swatp_nw_ies'
    pst_file = "swatp_nw_ies.pst"
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    # load prior simulation
    pr_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"swatp_nw_ies.0.obs.csv")
        )
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"swatp_nw_ies.{0}.obs.csv".format(9)))
    
    m_d2 = 'D:\\jj\\TxtInOut_Imsil_rye_rot_r2'
    org_sim = create_stf_sim_obd_df(m_d2, 1, "singi_obs_q1_colnam.csv", "cha01")

    par_obj_file = "swatp_nw_ies.9.par.objs.csv"
    '''

    df = create_stf_opt_df(pst, pt_oe)
    print(df)
    '''
    # obd_col = "obsval"
    # fig, ax = plt.subplots()
    # plot_stf_sim_obd(ax, df, obd_col)
    # plt.show()
    # single_plot_tseries_ensembles(pst, pr_oe, pt_oe, width=10, height=4, dot=False)

    # single_plot_fdc_added(pst, pr_oe, pt_oe, orgsim=org_sim, bstcs=["56", "171"])
    plot_par_obj(wd, pst, par_obj_file, objf="rsq", height=9)
    get_par_offset(pst)