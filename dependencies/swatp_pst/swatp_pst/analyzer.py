import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import pyemu
import os
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, FixedLocator
# from swatp_pst.handler import SWATp
from swatp_pst import objfns
import datetime
import scipy.stats as st
from scipy import stats
from swatp_pst import handler
from tqdm import tqdm


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
#     obs = obs.loc[obs.obgnme.apply(lambda x: x in psnnz_obs_groups),:]
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


# NOTE: will be deprecated
# def single_plot_fdc_added(
#                     pst, pr_oe, pt_oe, 
#                     width=10, height=8, dot=True,
#                     size=None, bstcs=None,
#                     orgsim=None
#                     ):
#     if size is None:
#         size = 30
#     # pst.try_parse_name_metadata()
#     # get the observation data from the control file and select 
#     obs = pst.observation_data.copy()
#     obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
#     time_col = []
#     for i in range(len(obs)):
#         time_col.append(obs.iloc[i, 0][-6:])
#     obs['time'] = time_col
#     onames = obs.obsnme.values

#     obs_d, obd_exd = convert_fdc_data(obs.obsval.values)
#     pr_min_d, pr_min_exd = convert_fdc_data(pr_oe.min().values)
#     pr_max_d, pr_max_exd = convert_fdc_data(pr_oe.max().values)
#     pt_min_d, pt_min_exd = convert_fdc_data(pt_oe.min().values)
#     pt_max_d, pt_max_exd = convert_fdc_data(pt_oe.max().values)

#     fig, ax = plt.subplots(figsize=(width,height))
#     ax.fill_between(pr_min_exd*100, pr_min_d, pr_max_d, interpolate=False, facecolor="0.5", alpha=0.4)
#     ax.fill_between(pt_min_exd*100, pt_min_d, pt_max_d, interpolate=False, facecolor="b", alpha=0.4)
#     ax.scatter(obd_exd*100, obs_d, color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
#     if orgsim is not None:
#         orgsim = orgsim
#         org_d, org_exd = convert_fdc_data(orgsim.iloc[:, 0].values)
#         ax.plot(org_exd*100, org_d, c='limegreen', lw=2, label="Original")
#     if bstcs is not None:
#         for bstc in bstcs:
#             dd, eexd = convert_fdc_data(pt_oe.loc[bstc,onames].values)
#             ax.plot(eexd*100, dd, lw=2, label=bstc)

#     ax.set_yscale('log')
#     ax.set_xlabel(r"Exceedence [%]", fontsize=12)
#     ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
#     ax.margins(0.01)
#     ax.tick_params(axis='both', labelsize=12)
#     plt.legend(fontsize=12, loc="lower left")
#     plt.tight_layout()
#     plt.savefig('fdc.png', bbox_inches='tight', dpi=300)
#     plt.show()
#     print(os.getcwd())  

#     # return pr_oe_min

def single_plot_fdc_added(
                    pst,
                    df,
                    obgnam,
                    width=10, height=8, dot=True,
                    size=None, bstc=False,
                    orgsim=None,
                    pr_oe=None
                    ):
    """plot flow exceedence

    :param df: dataframe created by get_pr_pt_df function
    :type df: dataframe
    :param width: figure width, defaults to 10
    :type width: int, optional
    :param height: figure hight, defaults to 8
    :type height: int, optional
    :param dot: scatter or line, defaults to True
    :type dot: bool, optional
    :param size: maker size, defaults to None
    :type size: int, optional
    :param bstcs: best candiates, defaults to None
    :type bstcs: list, optional
    :param orgsim: _description_, defaults to None
    :type orgsim: _type_, optional
    """
    df = df.loc[df["obgnme"]==obgnam]

    if size is None:
        size = 30
    obs_d, obd_exd = convert_fdc_data(df.obd.values)
    pr_min_d, pr_min_exd = convert_fdc_data(df.pr_min.values)
    pr_max_d, pr_max_exd = convert_fdc_data(df.pr_max.values)
    pt_min_d, pt_min_exd = convert_fdc_data(df.pt_min.values)
    pt_max_d, pt_max_exd = convert_fdc_data(df.pt_max.values)
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])
    # only non-zero observations
    # make a plot
    # ogs = obs.obgnme.unique()
    # ogs.sort()
    # print(ogs)

    # get values for x axis
    oobs = obs.loc[obs.obgnme==obgnam,:].copy()
    onames = oobs.obsnme.values
    fig, ax = plt.subplots(figsize=(width,height))
    if pr_oe:
        for i in pr_oe.index:
            # pr_oe_df = pr_oe.loc[i,onames].values
            pr_d, pr_exd = convert_fdc_data(pr_oe.loc[i,onames].values)
            ax.plot(
                    pr_exd*100, pr_d, "0.5",lw=1,alpha=0.5, 
                    label="Prior ensemble" if i == pr_oe.index[-1] else None,
                    zorder=1)
            # print(pr_oe_df)
    else:
        ax.fill_between(pr_min_exd*100, pr_min_d, pr_max_d, interpolate=False, facecolor="0.5", alpha=0.3, label="Prior ensemble")
    ax.fill_between(
        pt_min_exd*100, pt_min_d, pt_max_d, 
        interpolate=False, facecolor="g", alpha=0.6, label="Posterior ensemble",
        zorder=2)
    ax.scatter(
        obd_exd*100, obs_d, color='red',s=size, zorder=3, label="Observed", alpha=0.7, linewidths=0.5).set_facecolor("none")
    if orgsim is not None:
        orgsim = orgsim
        org_d, org_exd = convert_fdc_data(orgsim.iloc[:, 0].values)
        ax.plot(org_exd*100, org_d, c='limegreen', lw=2, label="Original")
    if bstc is True:
        # for bstc in bstcs:
        #     dd, eexd = convert_fdc_data(df.best_rel.values)
        #     ax.plot(eexd*100, dd, lw=2, label=bstc)
        # for bstc in bstcs:
        dd, eexd = convert_fdc_data(df.best_rel.values)
        ax.plot(eexd*100, dd, "b", lw=2, label="Best estimation",
                zorder=4)
    ax.set_yscale('log')
    ax.set_xlabel(r"Exceedence [%]", fontsize=12)
    ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
    ax.margins(0.01)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12, loc="lower left")
    plt.tight_layout()
    plt.savefig(f'fdc_{obgnam}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    print(os.getcwd())  




def convert_fdc_data(data):
    data = np.sort(data)[::-1]
    exd = np.arange(1.,len(data)+1) / len(data)
    return data, exd


def plot_tseries_ensembles_old(
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

def plot_tseries_ensembles(
                    pst, pr_oe, pt_oe, obgnam, width=10, height=3, 
                    dot=False, bstcd=None,
                    pt_fill=None,
                    ymin=None,
                    ymax=None,
                    # onames=["obd249lyr2"]
                    ):
    
    if pt_fill is not None:
        df = pt_fill.loc[pt_fill["obgnme"]==obgnam]
        print(df)
    # pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])
    # only non-zero observations
    # make a plot
    ogs = obs.obgnme.unique()
    ogs.sort()
    print(ogs)

    # get values for x axis
    oobs = obs.loc[obs.obgnme==obgnam,:].copy()
    # oobs.loc[:,"time"] = oobs.loc[:,"time"].astype(str)
#         oobs.sort_values(by="time",inplace=True)
    pr_oe = pd.DataFrame(pr_oe, index=pr_oe.index, columns=pr_oe.columns)
    pr_oe = pr_oe[(pr_oe > -999)]

    tvals = oobs.time.values
    onames = oobs.obsnme.values
    # obs['time'] = pd.to_datetime(obs['time'])
    fig,ax = plt.subplots(figsize=(width, height))


    if dot is True:
        # plot prior
        [ax.scatter(tvals,pr_oe.loc[i,onames].values,color="gray",s=30, alpha=0.5) for i in pr_oe.index]
        # plot posterior
        [ax.scatter(tvals,pt_oe.loc[i,onames].values,color='b',s=30,alpha=0.2) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        # tvals = oobs.time.values
        # onames = oobs.obsnme.values
        ax.scatter(oobs.time,oobs.obsval,color='red',s=30).set_facecolor("none")
    if dot is False:
        # plot prior
        [ax.plot(
            tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.6,
            label="Prior ensemble" if i == pr_oe.index[-1] else None,
            ) for i in pr_oe.index]
        # plot posterior
        if pt_fill is not None:
            ax.fill_between(
                df.index, df.pt_min, df.pt_max, interpolate=False, facecolor="g", alpha=0.6, label="Posterior ensemble",
                zorder=2)
        else:
            [ax.plot(tvals,pt_oe.loc[i,onames].values,"g",lw=0.5,alpha=0.7) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        # tvals = oobs.time.values
        # onames = oobs.obsnme.values
        ax.scatter(
            oobs.time,oobs.obsval,color='red',s=5, zorder=5, alpha=0.5,
            label="Observed"
            ).set_facecolor("none")
        # ax.scatter(oobs.time,oobs.obsval,color='red',s=1).set_facecolor("none")
        # ax.plot(oobs.time,oobs.obsval,"r-",lw=1)
    if bstcd is not None:
        ax.plot(tvals,pt_oe.loc[bstcd, onames].values,"b",lw=1, zorder=6, label="Best estimation")



    ax.tick_params(axis='x', labelrotation=90)
    ax.margins(x=0.01)
    # ax.set_title(og,loc="left")
    # fig.tight_layout()

    years = mdates.YearLocator()
    # print(years)
    yearsFmt = mdates.DateFormatter('%Y')  # add some space for the year label
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b') 
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthsFmt)
    plt.setp(ax.xaxis.get_minorticklabels(), fontsize=6, rotation=90)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    # ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"]*7)
    ax.tick_params(axis='both', labelsize=8, rotation=0)
    ax.tick_params(axis = 'x', pad=15)
    if ymin is not None:
        ax.set_ylim(ymin, ymax)
    # if obgnam
    # ax.set_ylabel("Stream Discharge $(m^3/s)$",fontsize=10)
    # ax.set_ylabel("Depth to water $(m)$",fontsize=14)
    # plt.legend(
    #     fontsize=10,
    #     ncol=4
    #     # loc="lower left"
    #     )
    plt.tight_layout()
    plt.savefig(f'tensemble_{obgnam}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    # '''

def plot_onetone_ensembles(
                    pst, pr_oe, pt_oe, width=5, height=4.5, dot=True,
                    dotsize=30
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
        [ax.scatter(oobs.obsval,pr_oe.loc[i,onames].values,color="gray",s=dotsize, alpha=0.5) for i in pr_oe.index]
        # plot posterior
        [ax.scatter(oobs.obsval,pt_oe.loc[i,onames].values,color='b',s=dotsize,alpha=0.2) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        # ax.scatter(oobs.obsval,oobs.obsval,color='red',s=dotsize).set_facecolor("none")
        ax.plot([0, 750], [0, 750], linestyle="--", color='k', alpha=0.3)

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
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(r"Monthly simulated irrigation $(mm/month)$", fontsize=12)
    ax.set_xlabel(r"Monthly measured irrigation $(mm/month)$", fontsize=12)
    ax.grid(True, alpha=0.5)
    # ax.margins(x=0.01)
    # ax.set_title(og,loc="left")
    plt.tight_layout()
    plt.savefig('onetoone.png', bbox_inches='tight', dpi=300)
    print(os.getcwd())
    plt.show()


def plot_prior_posterior_par_hist(
        wd,
        pst, prior_df, post_df, sel_pars, 
        width=7, height=5, ncols=3, bestcand=None, parobj_file=None):
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
                    color = "gray", alpha=0.5, density=False,
                    label="Prior"
            )
            y, x, _ = ax.hist(post_df.loc[:, colnam].values + offset,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20), 
                        alpha=0.5, density=False, label="Posterior"
            )
            ax.set_title(colnam, fontsize=9, ha='left', x=0.07, y=0.93, backgroundcolor='white')
            # ax.set_yticks([])
            if parobj_file is not None:
                po_df = pd.read_csv(os.path.join(wd, parobj_file))
                x = po_df.loc[po_df["real_name"]==bestcand, colnam].values + offset
                ax.axvline(x=x, color='r', linestyle="--", alpha=0.5)
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # ax.set_xticks(ax.get_xticks()[::1])
        
        ax.tick_params(axis='x', labelsize=8)       
    plt.ylabel(r"Frequency", fontsize=10)
    plt.xlabel(r"Parameter relative change (%)", fontsize=10)
    plt.tight_layout()
    plt.savefig('par_hist.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(os.getcwd())

def plot_flow_cal_val_hist(ax, flow_df, calidates, validates):
    cal_df = flow_df[calidates[0]:calidates[1]]
    val_df = flow_df[validates[0]:validates[1]]
    bin_min = flow_df.loc[:, ["flo_out", "obsval"]].values.min()
    bin_max = flow_df.loc[:, ["flo_out", "obsval"]].values.max()
    ax.hist(
        cal_df.loc[:, "flo_out"].values,
        bins=np.linspace(
            bin_min, bin_max, 50
        ), alpha=0.5, density=True,edgecolor='white')
    ax.hist(
        val_df.loc[:, "flo_out"].values,
        bins=np.linspace(
            bin_min, bin_max, 50
        ), alpha=0.5, density=True,edgecolor='white')    
    ax.hist(
        cal_df.loc[:, "obsval"].values,
        bins=np.linspace(
            bin_min, bin_max, 50
        ), alpha=0.5, density=True,edgecolor='white')
    ax.hist(
        val_df.loc[:, "obsval"].values,
        bins=np.linspace(
            bin_min, bin_max, 50
        ), alpha=0.5, density=True,edgecolor='white')

    print(val_df.mean())


def plot_wb_mon_cal_val_hist(ax, wd, colnam, calidates, validates):
    m1 = handler.SWATp(wd)
    start_day = m1.stdate_warmup
    df = m1.read_basin_wb_mon()
    df = df.loc[:, colnam]
    df.index = pd.date_range(start_day, periods=len(df), freq='ME')
    cal_df = df[calidates[0]:calidates[1]]
    val_df = df[validates[0]:validates[1]]
    bin_min = df.values.min()
    bin_max = df.values.max()

    ax.hist(
        cal_df.values,
        bins=np.linspace(
            bin_min, bin_max, 20
        ), alpha=0.5, density=True, edgecolor='white')
    ax.hist(
        val_df.values,
        bins=np.linspace(
            bin_min, bin_max, 20
        ), alpha=0.5, density=True,edgecolor='white')

def get_average_annual_wb(wd, colnam, calidates=None, validates=None):
    m1 = handler.SWATp(wd)
    start_day = m1.stdate_warmup
    df = m1.read_basin_wb_yr()
    df = df.loc[:, colnam]
    df.index = pd.date_range(start_day, periods=len(df), freq='YE') 
    if calidates is not None:   
        cal_df = df[calidates[0]:calidates[1]]
        val_df = df[validates[0]:validates[1]]
        print(cal_df.mean())
        print(val_df.mean())
    print(df.mean())


# data comes from hanlder module and SWATMFout class
def create_stf_sim_obd_df(wd, cha_id, obd_file, obd_col):
    m1 = handler.SWATp(wd)
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

def create_pcp_df(wd, cha_id):
    m1 = handler.SWATp(wd)
    start_day = m1.stdate_warmup
    sim = m1.read_pcp_data()
    sim = sim.loc[sim["gis_id"] == cha_id]
    sim["pcpmm"] = (sim["precip"]) /(sim["area"] *10000) *1000
    sim = sim.drop(['gis_id', 'precip', 'area'], axis=1)
    sim.index = pd.date_range(start_day, periods=len(sim.pcpmm), freq='ME')
    return sim  

    
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
    pcc = objfns.correlationcoefficient(obds, sims)
    return ns, pbias, rsq, rmse, mse, pcc

def get_rels_cal_val_objs(wd, pst_file, iter_idx=None, opt_idx=None, calval=None):
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

    if calval is None:
        calval = "cal"
    opt_df = opt_df.loc[opt_df["obgnme"]==calval]
    sims = opt_df.iloc[:, 0].tolist()
    obds = opt_df.iloc[:, 2].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    return ns, pbias, rsq, rmse

def get_rels_objs_new(df, obgnme=None):
    if obgnme is not None:
        df = df.loc[df["obgnme"]==obgnme]
    sims = df.loc[:, "best_rel"].tolist()
    obds = df.loc[:, "obd"].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    pcc = objfns.correlationcoefficient(obds, sims)
    return ns, pbias, rsq, rmse, mse, pcc

def get_p_factor(pst, pt_oe, perc_obd_nz=None, cal_val=False):
    """calculate p-factor

    :param pst: pst object
    :type pst: class
    :param pt_oe: posterior ensamble
    :type pt_oe: dataframe
    :param perc_obd_nz: percentage of observation noise, defaults to None
    :type perc_obd_nz: real, optional
    :param cal_val: option to separate calibration and validation, defaults to False
    :type cal_val: bool, optional
    :return: p-factor value
    :rtype: real
    """
    obs = pst.observation_data.copy()
    if perc_obd_nz is None:
        perc_obd_nz=10
    perc = perc_obd_nz*0.01
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])    
    df = pd.DataFrame(
        {'date':obs['time'],
        'obd':obs["obsval"],
        'weight':obs["weight"],
        'obgnme':obs["obgnme"],
        'pt_min': pt_oe.min(),
        'pt_max': pt_oe.max(),
        }
        )
    if cal_val is True:
        pfactors = []
        for i in ["cal", "val"]:
            cvdf = df.loc[df["obgnme"]==i]
            conditions = [
                ((cvdf.obd+(cvdf.obd*perc)) > cvdf.pt_min) & 
                ((cvdf.obd-(cvdf.obd*perc)) < cvdf.pt_max)
                    ]
            cvdf['pfactor'] = np.select(
                conditions, [1], default=0
                )
            pfactor = cvdf.loc[:, 'pfactor'].value_counts()[1] / len(cvdf.loc[:, 'pfactor'])
            pfactors.append(pfactor)
        print(pfactors)
        return pfactors
    else:
        conditions = [
            ((df.obd+(df.obd*perc)) > df.pt_min) & 
            ((df.obd-(df.obd*perc)) < df.pt_max)
                ]
        df['pfactor'] = np.select(
            conditions, [1], default=0
            )
        pfactor = df.loc[:, 'pfactor'].value_counts()[1] / len(df.loc[:, 'pfactor'])
        print(pfactor)
        df.to_csv('testpfactor.csv')
        return pfactor
    

def get_d_factor(pst, pt_oe, cal_val=False):
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])    
    df = pd.DataFrame(
        {'date':obs['time'],
        'obd':obs["obsval"],
        'weight':obs["weight"],
        'obgnme':obs["obgnme"],
        'pt_min': pt_oe.min(),
        'pt_max': pt_oe.max(),
        }
        )
    if cal_val is True:
        dfactors = []
        for i in ["cal", "val"]:
            cvdf = df.loc[df["obgnme"]==i]
            std_obd = np.std(cvdf['obd'])
            dist_pts = (cvdf['pt_max'] - cvdf['pt_min']).mean()
            dfactor = dist_pts/std_obd
            dfactors.append(dfactor)
        print(dfactors)
        return dfactors
    else:
        std_obd = np.std(df['obd'])
        dist_pts = (df['pt_max'] - df['pt_min']).mean()
        dfactor = dist_pts/std_obd
        print(dfactor)
        return dfactor


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
    mses = []
    pccs = []
    # for i in range(np.shape(pt_oe)[0]):
    for i in pt_oe.index:
        ns, pbias, rsq, rmse, mse, pcc = get_rels_objs(wd, pst_file, iter_idx=iter_idx, opt_idx=i)
        nss.append(ns)
        pbiass.append(pbias)
        rsqs.append(rsq)
        rmses.append(rmse)
        mses.append(mse)
        pccs.append(pcc)
    objs_df = pd.DataFrame(
        {
            "ns": nss, "pbias": pbiass, "rsq": rsqs, "rmse": rmses,
            "mse": mses, "pcc":pccs
            },
        index=pt_oe.index)
    pt_oe_df = pd.concat([pt_oe_df, objs_df], axis=1)
    pt_par_df = pd.concat([pt_par_df, objs_df], axis=1)
    pt_oe_df.to_csv(os.path.join(wd, "{0}.{1}.obs.objs.csv".format(pst_nam, iter_idx)))
    pt_par_df.to_csv(os.path.join(wd, "{0}.{1}.par.objs.csv".format(pst_nam, iter_idx)))

def filter_candidates(
        wd, pst, par_obj_file, parbds=None,
        nsbds=None, pbiasbds=None,
        rsqbds=None, rmsebds=None,
        savefile=False):
    pst_nam = par_obj_file[:-4]
    pars_info = get_par_offset(pst)
    po_df = pd.read_csv(os.path.join(wd, par_obj_file))
    if parbds is not None:
        for parnam in pars_info.parnme:
            po_df = po_df.query(f"{parnam}>={parbds[0]} & {parnam}<={parbds[1]}")
    if nsbds is not None:
        po_df = po_df.loc[(po_df["ns"]>=nsbds[0]) & (po_df["ns"]<=nsbds[1])]
    if pbiasbds is not None:
        po_df = po_df.query(f"pbias>={pbiasbds[0]} & pbias<={pbiasbds[1]}")
    if rsqbds is not None:
        po_df = po_df.loc[(po_df["rsq"]>=rsqbds[0]) & (po_df["rsq"]<=rsqbds[1])]
    if rmsebds is not None:
        po_df = po_df.loc[(po_df["rmse"]>=rmsebds[0]) & (po_df["rmse"]<=rmsebds[1])]
    if savefile is True:
        po_df.to_csv(os.path.join(wd, "{}.filter.csv".format(pst_nam)), index=False)
    print(po_df)
    return po_df



def get_par_offset(pst):
    pars = pst.parameter_data.copy()
    pars = pars.loc[:, ["parnme", "offset"]]
    return pars
    

def plot_par_obj(
        wd, pst,  par_obj_file, 
        objf=None, width=7, height=3, ncols=3,
        bstcs=None, orgsim=None,
        save_fig=False):
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
            if bstcs is not None:
                for bstc, colr in zip(bstcs, ["blue"]):
                    x = po_df.loc[po_df["real_name"]==bstc, par_cols[i]].values + offset
                    print(x)
                    ax.axvline(x=x, color=colr, linestyle="--", alpha=0.5)
        # if bstcs is not None:
            # ax.set_yticks([])
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)   
        if orgsim is not None:
            orgsim = orgsim
    plt.xlabel("Parameter relative change (%)")
    plt.tight_layout()

    if save_fig is True:
        plt.savefig(
            os.path.join(wd, f'par_obj_{objf}.png'), 
            bbox_inches='tight', dpi=300)
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


def get_pr_pt_df(pst, pr_oe, pt_oe, bestrel_idx=None):
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])
    # print(pt_oe.loc["4"])
    if bestrel_idx is not None:
        df = pd.DataFrame(
            {'date':obs['time'],
            'obd':obs["obsval"],
            'pr_min': pr_oe.min(),
            'pr_max': pr_oe.max(),
            'pt_min': pt_oe.min(),
            'pt_max': pt_oe.max(),
            'best_rel': pt_oe.loc[bestrel_idx],
            'obgnme': obs['obgnme'],
            }
            )
    else:
        df = pd.DataFrame(
            {'date':obs['time'],
            'obd':obs["obsval"],
            'pr_min': pr_oe.min(),
            'pr_max': pr_oe.max(),
            'pt_min': pt_oe.min(),
            'pt_max': pt_oe.max(),
            'obgnme': obs['obgnme'],}
            )
    df.set_index('date', inplace=True)
    return df

def plot_fill_between_ensembles(
        df, 
        width=12, height=4,
        caldates=None,
        valdates=None,
        size=None,
        pcp_df=None,
        bestrel_idx=None,
        ):
    """plot time series of prior/posterior predictive uncertainties

    :param df: dataframe of prior/posterior created by get_pr_pt_df function
    :type df: dataframe
    :param width: plot width, defaults to 12
    :type width: int, optional
    :param height: plot height, defaults to 4
    :type height: int, optional
    :param caldates: calibration start and end dates, defaults to None
    :type caldates: list, optional
    :param valdates: validation start and end dates, defaults to None
    :type valdates: list, optional
    :param size: symbol size, defaults to None
    :type size: int, optional
    :param pcp_df: dataframe of precipitation, defaults to None
    :type pcp_df: dataframe, optional
    :param bestrel_idx: realization index, defaults to None
    :type bestrel_idx: string, optional
    """
    if size is None:
        size = 30
    fig, ax = plt.subplots(figsize=(width,height))
    x_values = df.loc[:, "newtime"].values
    # x_values = df.index.values
    if caldates is not None:
        caldf = df[caldates[0]:caldates[1]]
        valdf = df[valdates[0]:valdates[1]]
        ax.fill_between(
            df.index.values, df.loc[:, 'pr_min'].values, df.loc[:, 'pr_max'].values, 
            facecolor="0.5", alpha=0.4, label="Prior")
        ax.fill_between(
            caldf.index.values, caldf.loc[:, 'pt_min'].values, caldf.loc[:, 'pt_max'].values, 
            facecolor="g", alpha=0.4, label="Posterior")
        ax.plot(caldf.index.values, caldf.loc[:, 'best_rel'].values, c='g', lw=1, label="calibrated")
        ax.fill_between(
            valdf.index.values, valdf.loc[:, 'pt_min'].values, valdf.loc[:, 'pt_max'].values, 
            facecolor="m", alpha=0.4, label="Forecast")        
        ax.scatter(
            df.index.values, df.loc[:, 'obd'].values, 
            color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
        ax.plot(valdf.index.values, valdf.loc[:, 'best_rel'].values, c='m', lw=1, label="validated")
    else:
        ax.fill_between(
            x_values, df.loc[:, 'pr_min'].values, df.loc[:, 'pr_max'].values, 
            facecolor="0.5", alpha=0.4, label="Prior")
        ax.fill_between(
            x_values, df.loc[:, 'pt_min'].values, df.loc[:, 'pt_max'].values, 
            facecolor="g", alpha=0.4, label="Posterior")
        ax.plot(x_values, df.loc[:, 'best_rel'].values, c='g', lw=1, label="calibrated")
        ax.scatter(
            x_values, df.loc[:, 'obd'].values, 
            color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
    if pcp_df is not None:
        # pcp_df.index.freq = None
        ax2=ax.twinx()
        ax2.bar(
            pcp_df.index, pcp_df.loc[:, "pcpmm"].values, label='Precipitation',
            width=20 ,
            color="blue", 
            align='center', 
            alpha=0.5
            )
        ax2.set_ylabel("Precipitation $(mm/month)$",fontsize=12)
        ax2.invert_yaxis()
        ax2.set_ylim(pcp_df.loc[:, "pcpmm"].max()*3, 0)
        # ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)
    # ax.axvline(datetime.datetime(2016,12,31), linestyle="--", color='k', alpha=0.3)
    # ax.set_xlabel(r"Exceedence [%]", fontsize=12)
    ax.set_ylabel(r"Monthly irrigation $(mm/month)$", fontsize=12)
    # ax.set_ylabel(r"Monthly streamflow $(m^3/s)$", fontsize=12)
    ax.margins(0.01)
    ax.tick_params(axis='both', labelsize=12)
    # ax.set_ylim(0, df.max().max()*1.5)
    ax.set_ylim(0, 800)
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    order = [0,1,2,3]

    tlables = labels
    tlines = lines
    '''
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    '''
    fig.legend(
        [tlines[idx] for idx in order],[tlables[idx] for idx in order],
        fontsize=10,
        loc = 'lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncols=7)
    # fig.legend(fontsize=12, loc="lower left")
    # years = mdates.YearLocator()
    # print(years)
    # yearsFmt = mdates.DateFormatter('%Y')  # add some space for the year label
    # months = mdates.MonthLocator()
    # monthsFmt = mdates.DateFormatter('%b') 
    # ax.xaxis.set_minor_locator()
    # ax.xaxis.set_minor_formatter(monthsFmt)
    # plt.setp(ax.xaxis.get_minorticklabels(), fontsize = 8, rotation=90)
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(yearsFmt)
    ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"]*7)
    ax.tick_params(axis='both', labelsize=8, rotation=90)
    ax.tick_params(axis = 'x', pad=20)

    plt.tight_layout()
    plt.savefig('cal_val.png', bbox_inches='tight', dpi=300)
    plt.show()




# sensitivity
def read_sobol_sti(wd, pst_file):
    return pd.read_csv(
        os.path.join(wd, f"{pst_file[:-4]}.sobol.sti.csv")
        )

def read_sobol_sfi(wd, pst_file):
    return pd.read_csv(
        os.path.join(wd, f"{pst_file[:-4]}.sobol.si.csv")
        )

def confidence_interval(data):
    cfi = st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    error = cfi[1] - cfi[0]
    return error

def plot_sen_sobol(wd, pst_file):
    # let's get zero for negative values
    # st
    sfdf = read_sobol_sfi(wd, pst_file)
    sfphi = sfdf.loc[sfdf["output"]=="phi"]
    sfphi[sfphi.iloc[:, 1:]<0] = 0
    sfdf[sfdf.iloc[:, 1:]<0] = 0
    sfts = sfdf.iloc[1:, 1:]
    sftsm = sfts.mean()
    stdf = read_sobol_sti(wd, pst_file)
    stphi = stdf.loc[stdf["output"]=="phi"] # phi value
    stphi[stphi.iloc[:, 1:]<0] = 0 # get zero for negative values
    stdf[stdf.iloc[:, 1:]<0] = 0 # get zero for negative values
    stts = stdf.iloc[1:, 1:]
    sttsm = stts.mean()
    sfts_cfis = []
    for par in sfphi.columns[1:]:
        sfts_cfis.append(confidence_interval(sfts.loc[:, par].values))
    stts_cfis = []
    for par in stphi.columns[1:]:
        stts_cfis.append(confidence_interval(stts.loc[:, par].values))    
    N = len(sfphi.columns[1:])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4
    # phiwidth = 0.2
    

    fig, ax = plt.subplots(figsize=(12,4))
    # Width of a bar 
    error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5)

    tcolor = sftsm + sttsm
    colors = plt.cm.rainbow(tcolor/max(tcolor))
    tphicolor = sfphi.iloc[0, 1:].values + stphi.iloc[0, 1:].values
    # phico = plt.cm.rainbow(tphicolor/max(tphicolor))
    # ax.plot(np.NaN, np.NaN, '-', color='none', label='Variance based')

    ax.bar(
        ind, sftsm, width, 
        color="C0", yerr=sfts_cfis, label=r"First order $S_i$", 
        error_kw=error_kw
        )
    ax.bar(
        ind + width, sttsm, width,
        color="C1", yerr=stts_cfis, label=r"Total order $S_{Ti}$", error_kw=error_kw,
        )
    # ax.plot(np.NaN, np.NaN, '-', color='none', label=r'Objective function $(phi)$')
    # ax.bar(
    #     ind + width, sfphi.iloc[0, 1:].values, width, 
    #     color="C1", label=r"First order $S_i$ - objective function",
    #     )
    # ax.bar(
    #     ind + width, stphi.iloc[0, 1:].values, width, 
    #     bottom=sfphi.iloc[0, 1:].values,
    #     color='C1', label=r"Total order $S_{Ti} - objective function$", alpha=0.5
    #     )

    # ax.bar(ind + width, stphi.iloc[0, 1:].abs().values, width, color='C0', yerr=stts_cfis, label="Total order", error_kw=error_kw)
    # ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(sfphi.columns[1:])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(r"Sensitivity index", fontsize=12)
    ax.set_xlabel(r"Parameter", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    # ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.tick_params(
        which="both",
        axis="y",direction="in", 
        # pad=-22
        )
    # ax.grid('True')
    plt.margins(y=0.1) 
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'sen_sobol.png'), bbox_inches='tight', dpi=300)
    plt.show()

    # '''

def plot_sen_sobol2(wd, pst_file):
    # let's get zero for negative values
    # st
    sfdf = read_sobol_sfi(wd, pst_file)
    sfphi = sfdf.loc[sfdf["output"]=="phi"]
    sfphi[sfphi.iloc[:, 1:]<0] = 0
    sfdf[sfdf.iloc[:, 1:]<0] = 0
    sfts = sfdf.iloc[1:, 1:]
    sftsm = sfts.mean()
    stdf = read_sobol_sti(wd, pst_file)
    stphi = stdf.loc[stdf["output"]=="phi"] # phi value
    stphi[stphi.iloc[:, 1:]<0] = 0 # get zero for negative values
    stdf[stdf.iloc[:, 1:]<0] = 0 # get zero for negative values
    stts = stdf.iloc[1:, 1:]
    sttsm = stts.mean()
    sfts_cfis = []
    for par in sfphi.columns[1:]:
        sfts_cfis.append(confidence_interval(sfts.loc[:, par].values))
    stts_cfis = []
    for par in stphi.columns[1:]:
        stts_cfis.append(confidence_interval(stts.loc[:, par].values))    
    N = len(sfphi.columns[1:])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4
    phiwidth = 0.2
    fig, ax = plt.subplots(figsize=(12,4))
    # Width of a bar 
    error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5)

    tcolor = sftsm + sttsm
    colors = plt.cm.rainbow(tcolor/max(tcolor))
    tphicolor = sfphi.iloc[0, 1:].values + stphi.iloc[0, 1:].values
    # phico = plt.cm.rainbow(tphicolor/max(tphicolor))
    # ax.plot(np.NaN, np.NaN, '-', color='none', label='Variance based')

    ax.bar(
        ind, sftsm, width, 
        color="C0", yerr=sfts_cfis, label=r"First order $S_i - variance$", 
        error_kw=error_kw
        )
    ax.bar(
        ind, sttsm, width, bottom=sftsm,
        color="C0", yerr=sfts_cfis, label=r"Total order $S_{Ti} - variance$", error_kw=error_kw,
        alpha=0.5
        )
    # ax.plot(np.NaN, np.NaN, '-', color='none', label=r'Objective function $(phi)$')
    ax.bar(
        ind + 0.3, sfphi.iloc[0, 1:].values, phiwidth, 
        color="C1", label=r"First order $S_i$ - objective function",
        )
    ax.bar(
        ind + 0.3, stphi.iloc[0, 1:].values, phiwidth, 
        bottom=sfphi.iloc[0, 1:].values,
        color='C1', label=r"Total order $S_{Ti} - objective function$", alpha=0.5
        )

    # ax.bar(ind + width, stphi.iloc[0, 1:].abs().values, width, color='C0', yerr=stts_cfis, label="Total order", error_kw=error_kw)
    # ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(sfphi.columns[1:])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(r"Sensitivity index", fontsize=12)
    ax.set_xlabel(r"Parameter", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    # ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.tick_params(
        which="both",
        axis="y",direction="in", 
        # pad=-22
        )
    # ax.grid('True')
    plt.margins(y=0.1) 
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'sen_sobol.png'), bbox_inches='tight', dpi=300)
    plt.show()



def get_sobol_results(wd, pstfile):
    #first
    sfdf = read_sobol_sfi(wd, pstfile)
    sfdf[sfdf.iloc[:, 1:]<0] = 0
    sfts = sfdf.iloc[1:, 1:]
    sftsm = sfts.mean()
    stdf = read_sobol_sti(wd, pstfile)
    stdf[stdf.iloc[:, 1:]<0] = 0 # get zero for negative values
    stts = stdf.iloc[1:, 1:]
    sttsm = stts.mean()
    sfts_cfis = []
    for par in sftsm.index:
        sfts_cfis.append(confidence_interval(sfts.loc[:, par].values))
    stts_cfis = []
    for par in sttsm.index:
        stts_cfis.append(confidence_interval(stts.loc[:, par].values))    
    return sftsm, sttsm, sfts_cfis, stts_cfis


    # for par in sfphi.columns[1:]:
    #     sfts_cfis.append(confidence_interval(sfts.loc[:, par].values))
    # stts_cfis = []
    # for par in stphi.columns[1:]:
    #     stts_cfis.append(confidence_interval(stts.loc[:, par].values))  

def plot_sen_sobol_jj(wd, pstfile, wd2, pstfile2):
    sftsm, sttsm, sfts_cfis, stts_cfis = get_sobol_results(wd, pstfile)
    sftsm2, sttsm2, sfts_cfis2, stts_cfis2 = get_sobol_results(wd2, pstfile2)

    N = len(sttsm)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.6
    # phiwidth = 0.2
    fig, axes = plt.subplots(ncols=2, figsize=(7,7), gridspec_kw={'width_ratios': [4, 1]})
    # ax1 = fig.add_subplot(111, frameon=False)
    # ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # Width of a bar 
    error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5)
    axes[0].barh(
        ind, -sftsm, width, 
        color="C0", xerr=sfts_cfis, label=r"First order $S_i$", 
        error_kw=error_kw
        )
    axes[0].barh(
        ind, -sttsm, width,left=-sftsm,
        color="C0", xerr=stts_cfis, label=r"Total order $S_{Ti}$", error_kw=error_kw,
        alpha=0.5
        )
    axes[0].barh(
        ind, sftsm2, width, 
        color="C1", xerr=sfts_cfis2, label=r"First order $S_i$", 
        error_kw=error_kw
        )
    axes[0].barh(
        ind, sttsm2, width, left=sftsm2,
        color="C1", xerr=stts_cfis2, label=r"Total order $S_{Ti}$", error_kw=error_kw,
        alpha=0.5
        )

    axes[1].barh(
        ind, sftsm2, width, 
        color="C1", xerr=sfts_cfis2, label=r"First order $S_i$", 
        error_kw=error_kw
        )
    axes[1].barh(
        ind, sttsm2, width, left=sftsm2,
        color="C1", xerr=stts_cfis2, label=r"Total order $S_{Ti}$", error_kw=error_kw,
        alpha=0.5
        )
    axes[0].set_yticks(ind)
    axes[0].set_yticklabels(sftsm.index)
    fig.supxlabel(r"Sensitivity index", fontsize=12)
    # axes[0].set_xlabel(r"Sensitivity index1", fontsize=12)
    axes[0].set_ylabel(r"Parameter", fontsize=12)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=12)
    # axes[0].legend(fontsize=10, loc="upper left")
    # axes[0].yaxes[0]is.get_ticklocs(minor=True)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        ax.tick_params(
            which="both",
            axis="x",direction="in", 
            # pad=-22
            )
    # ax.grid('True')
    # plt.margins(y=0.1) 
    axes[1].set_xlim(0.7,3.7)
    axes[1].get_yaxis().set_visible(False)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[1].spines[['left', 'top', 'right']].set_visible(False)
    d = 1.3  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot(1, 0 , transform=axes[0].transAxes, **kwargs)
    # axes[0].plot([1, 0], [1, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot(0, 0, transform=axes[1].transAxes, **kwargs)

    # axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # xtlabels = [range(-1)]

    axes[0].set_xlim(-1.2, 0.7)
    axes[0].xaxis.set_major_locator(FixedLocator([-1.0, -0.5, 0, 0.5]))
    axes[0].set_xticklabels(abs(axes[0].get_xticks()))

    # axes[0].set_xticklabels([f"{x}" for x in [1.0, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1.0]])

    # axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1].xaxis.set_tick_params(which='minor', bottom=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(wd, 'sen_sobol_jj.png'), bbox_inches='tight', dpi=300)
    plt.show()

def jj_paper(wd1, wd2, pst_file1, pst_file2):
    sfdf1 = read_sobol_sfi(wd1, pst_file1)
    sfdf1[sfdf1.iloc[:, 1:]<0] = 0
    sfts1 = sfdf1.iloc[1:, 1:]
    sftsm1 = sfts1.mean()
    stdf1 = read_sobol_sti(wd1, pst_file1)
    stdf1[stdf1.iloc[:, 1:]<0] = 0 # get zero for negative values
    stdf1[stdf1.iloc[:, 1:]<0] = 0 # get zero for negative values
    stts1 = stdf1.iloc[1:, 1:]
    sttsm1 = stts1.mean()


def albufera_predictive_results(wd):
    colnam = 'et'
    pstfile = "alb_rw_ies.pst"
    iter_idx = 11
    # get_average_annual_wb(wd, colnam)
    pst = pyemu.Pst(os.path.join(wd, pstfile))
    pst_nam = pstfile[:-4]
    pr_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, 0)))
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))
    # get_p_factor(pst, pt_oe)
    # get_d_factor(pst, pt_oe)
    # plot_onetone_ensembles(pst, pr_oe, pt_oe, dotsize=15)
    # single_plot_tseries_ensembles(pst, pr_oe, pt_oe, dot=True)
    dff = get_pr_pt_df(pst, pr_oe, pt_oe)
    # dff['newtime'] =dff.index.strftime('%Y-\n%b')
    print(dff)
    # plot_fill_between_ensembles(dff)
    single_plot_fdc_added(pst, dff, "cha049", pr_oe=pr_oe)





    # sfdf2 = read_sobol_sfi(wd2, pst_file2)



class SWATp(handler.Paddy):
    def __init__(self, wd):
        super().__init__(wd)
        os.chdir(wd)

    def plot_stress(self, df, stress=None, w=12, h=4):
        """plot stress for crop

        :param df: dataframe from handler.get_paddy_stress_df
        :type df: dataframe
        :param stress: stress type, defaults to None, strsw
        :type stress: string, optional
        :return: bar charts for stress
        :rtype: figre
        """
        if stress is None:
            stress = "strsw"

        N = len(df.columns)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.4

        fig, ax = plt.subplots(figsize=(w,h))
        # Width of a bar 
        error_kw=dict(lw=1, capsize=2, capthick=1, alpha=0.5)

        # tcolor = sftsm + sttsm
        # colors = plt.cm.rainbow(tcolor/max(tcolor))
        # tphicolor = sfphi.iloc[0, 1:].values + stphi.iloc[0, 1:].values
        # # phico = plt.cm.rainbow(tphicolor/max(tphicolor))
        # # ax.plot(np.NaN, np.NaN, '-', color='none', label='Variance based')

        # ax.bar(
        #     df.index, df.strsw, width, 
        #     color="C0", label=r"First order $S_i - variance$", 
        #     error_kw=error_kw
        #     )
        ax.bar(
            df.index, df.loc[:, stress], width, 
            color="C0", label=r"First order $S_i - variance$", 
            error_kw=error_kw
            )      

        # ax.bar(
        #     ind, sttsm, width, bottom=sftsm,
        #     color="C0", yerr=sfts_cfis, label=r"Total order $S_{Ti} - variance$", error_kw=error_kw,
        #     alpha=0.5
        #     )
        # # ax.plot(np.NaN, np.NaN, '-', color='none', label=r'Objective function $(phi)$')
        # ax.bar(
        #     ind + 0.3, sfphi.iloc[0, 1:].values, phiwidth, 
        #     color="C1", label=r"First order $S_i$ - objective function",
        #     )
        # ax.bar(
        #     ind + 0.3, stphi.iloc[0, 1:].values, phiwidth, 
        #     bottom=sfphi.iloc[0, 1:].values,
        #     color='C1', label=r"Total order $S_{Ti} - objective function$", alpha=0.5
        #     )
        ax.margins(x=0.01)
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        plt.savefig(f'stress_{stress}.png', bbox_inches='tight', dpi=300)
        plt.show()
        print(os.getcwd())


# '''
class Paddy(handler.Paddy):
    def __init__(self, wd):
        super().__init__(wd)
        os.chdir(wd)

    def plot_paddy_daily(self, df):
        cmap = plt.get_cmap("tab10")
        nums = len(df.columns)
        fig, axes = plt.subplots(nrows=nums, sharex=True, figsize=(9, 11))
        for i, (col, ax) in enumerate(zip(df.columns, axes)):
            ax.plot(df.index, df[col], color=cmap(i), label=col)
            ax.legend(loc="upper left", fontsize=12)
        
            ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

    def plot_yield(self, df):
        fig, ax = plt.subplots()
        ax.plot(df.index, df["yield"], "v-",markerfacecolor="None", label="Simulated, Botanga HRU Model")
        ax.plot(df.index, df["obd_yield"], "o-", markerfacecolor="None", label="Observed, District Data")
        ax.tick_params(axis='both', labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_prep(self, df):
        fig, ax = plt.subplots()
        ax.plot(df.index, df["precip"], "v-",markerfacecolor="None", label="CHIRPS, Botanga HRU Model")
        ax.plot(df.index, df["northern"], "o-", markerfacecolor="None", label="Observed, Northern Regional Data")
        ax.tick_params(axis='both', labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


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


    def plot_violin2(self, inf, month, wd=None, cropBHU=None, width=None, height=None):
        if wd is None:
            wd = os.getcwd()
        if cropBHU is None:
            cropBHU = 0
        # Boxplot
        # f, ax = plt.subplots(3, 4, figsize=(12,8), sharex=True, sharey=True)
        days = handler.Paddy(wd).heatunit_days(inf, month, cropBHU=cropBHU)
        if width is not None:
            f, axes = plt.subplots(
                nrows=1, ncols=len(days), figsize=(width,height), sharey=True
                )
        else:
            f, axes = plt.subplots(nrows=1, ncols=len(days), sharey=True)
        x_names = [str(i) for  i in days]
        # plot. Set color of marker edge
        flierprops = dict(
                        marker='o', 
                        markerfacecolor='#fc0384', 
                        markersize=7,
                        # linestyle='None',
                        # markeredgecolor='none',
                        alpha=0.3)
        # ax.boxplot(data, flierprops=flierprops)
        # os.chdir(wd)

        for ax, day in tqdm(zip(axes, days), total=len(days)):
            df = handler.Paddy(wd).generate_heatunit(inf, month, day, cropBHU=cropBHU)
            r = ax.violinplot(
                df.loc[:, "FPHU0"].values,  
                widths=(0.5),
                showmeans=True, showextrema=True, showmedians=False,
                # quantiles=[[0.25, 0.75]]*len(days),
                quantiles=[[0.25, 0.75]],
                bw_method='silverman'
                )
            r['cmeans'].set_color('r')
            r['cquantiles'].set_color('r')
            r['cquantiles'].set_linestyle(':')
            # r['cquantiles'].set_linewidth(3)
            # colors = ['#c40243', "#04b0db", '#038f18', ]
            # for c, pc in zip(colors, r['bodies']):
            #     pc.set_facecolor(c)
            # #     pc.set_edgecolor('black')
            #     pc.set_alpha(0.4)

            ax.set_xticks([1])
            # ax.set_xticklabels(df_m.keys(), rotation=90)
            ax.set_xticklabels([str(day)])
            # ax.set_xticklabels(x_names)
            ax.tick_params(axis='both', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y')
            
            # ax.set_xticklabels([str(day)])
        # ax.spines['bottom'].set_visible(False)
        # ax.set_ylabel('CH$_4$ emission $(g\;CH_{4}-C\; m^{-2}\cdot d^{-1})$', fontsize=14)
        # plt.xticks([0])
        plt.tight_layout()
        lastfolder = os.path.basename(os.path.normpath(os.getcwd()))
        plt.savefig(os.path.join(os.getcwd(), f'HUI_{lastfolder}.png'), dpi=300, bbox_inches="tight")
        plt.show()






# '''

if __name__ == '__main__':

    # NOTE: paddy convert
    wd =  "D:\\Projects\\Watersheds\\Albufera\\2nd\\alb_rw_ies"
    albufera_predictive_results(wd)
    
