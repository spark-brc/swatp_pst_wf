import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import spline

#  Working directories ==========================================
mwd = ("d:\\SG_papers\\2018_Park_et_al_QuantifyingWaterUsingSM\\scripts")
d025 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d025"
d05 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d05"
d075 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d075"
d125 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d125"
d15 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d15"
d175 = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\d175"
org = "d:\\projects\\MiddleBosque\\climate_scenarios\\drought\\org"

startDate = '1/1/1980'

wds = [d025, d05, d075, d125, d15, d175, org]
datasets = ["d025", "d05", "d075", "d125", "d15", "d175", "org"]

ds2 = ["d025", "d05", "d075", "d125", "d15", "d175"]
colors = (
    'brown', 'chocolate', 'gold',
    'greenyellow', 'limegreen', 'forestgreen',)


def flood():
    pcp = "pcp1.pcp"
    wds = [d_127, d_254, d_381, d_508, w_127, w_254, w_381, w_508, org]
    datasets = ["d_127", "d_254", "d_381", "d_508", "w_127", "w_254", "w_381", "w_508", "org"]

    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    pcpdf = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        with open(os.path.join(wd, pcp), "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        pdata = [float(x[7:12]) for x in data] # Only data

        add_pcpdf = pd.DataFrame({ds: pdata})
        pcpdf = pd.concat([pcpdf, add_pcpdf[ds]], axis=1)
    pcpdf.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    dff = pcpdf['2004-7-27':'2004-8-1']
    wdff = pcpdf['2004-7-28':'2004-7-30']
    # print(wdff)
    # dff.replace(-99.0, np.nan, inplace=True)
    # s1 = pd.Series(pd.factorize(dff.index)[0] + 1, dff.index)
    # x_smooth = np.linspace(s1.min(), s1.max(), 100) #300 represents number of points to make between T.min and T.max
    # y_smooth = spline(s1.values, dff.d_508, x_smooth)

    fig, ax = plt.subplots()
    # plt.plot(x_smooth, y_smooth)
    ax.bar(dff.index, (dff["d_508"]/dff["d_508"].max()*-1), color='indigo')
    ax.bar(dff.index, (dff["d_381"]/dff["d_508"].max()*-1), color='darkviolet')    
    ax.bar(dff.index, (dff["d_254"]/dff["d_508"].max()*-1), color='mediumorchid')   
    ax.bar(dff.index, (dff["d_127"]/dff["d_508"].max()*-1), color='violet')
    ax.bar(dff.index, (dff["w_508"]/dff["w_508"].max()), color='indigo')
    ax.bar(dff.index, (dff["w_381"]/dff["w_508"].max()), color='darkviolet')    
    ax.bar(dff.index, (dff["w_254"]/dff["w_508"].max()), color='mediumorchid')   
    ax.bar(dff.index, (dff["w_127"]/dff["w_508"].max()), color='violet')
    barlist = ax.bar(wdff.index, (wdff["w_127"]/dff["w_508"].max()), color='royalblue')
    barlist[0].set_color('skyblue')
    barlist[2].set_color('lightskyblue')    

    # moving bottom spine up to y=0 position:
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    xlims = ax.get_xlim()
    ax.set_xticks(xlims)
    # ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top='off', bottom='on', left='off', right='off')
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels(["Jul\n2004", "Aug\n2004"])

    # Legend
    names = (
        '$P_{20}$', '$P_{15}$', '$P_{10}$', '$P_{5}$',
        '$P_{2.4}$', '$P_{0.5 - 0.6}$')
    colors = (
        'indigo', 'darkviolet', 'mediumorchid', 'violet', 'royalblue', 'lightskyblue')
    ps = []
    for c in colors:
        ps.append(Rectangle((0, 0), width=0.1, height=0.125, fc=c, alpha=1))
    legend = ax.legend(
        ps, names,
        loc='upper left',
        title="EXPLANATION",
        edgecolor='none',
        fontsize=8,
        bbox_to_anchor=(-0.05, 1),
        ncol=1)
    legend._legend_box.align = "left"

    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)

    # # Annotation
    # ax.annotate(
    #     '', xy=(0.2, 0.65), xycoords=ax.transAxes,
    #     xytext=(0.2, .35), textcoords=ax.transAxes,
    #     horizontalalignment='center',
    #     arrowprops=dict(color='grey', arrowstyle='-'))

    # ax.annotate(
    #     'Flood', xy=(0.575, 0.88), xycoords=ax.transAxes,
    #     xytext=(0.575, 1.05), textcoords=ax.transAxes,
    #     horizontalalignment='center',
    #     arrowprops=dict(arrowstyle='<|-'))

    ax.text(.39, .6, 'Wet', style='italic', transform=ax.transAxes)
    ax.text(.39, .4, 'Dry', style='italic', transform=ax.transAxes)


    plt.show()


def streamflow_m():
    rchN, colN = 58, 6
    sDay = "1/1/1985"
    sDay_cali = "12/1/1998"
    eDay_cali = "12/1/2000"

    # Open file for loop
    
    dff = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        output = pd.read_csv(
                            os.path.join(wd, "output.rch"),
                            delim_whitespace=True,
                            skiprows=9,
                            usecols=[1, 3, colN],
                            names=["date", "filter", ds],
                            index_col=0)   
        dfm = output.loc[rchN]
        dfm = dfm[dfm['filter'] < 13]
        dfm.index = pd.date_range(sDay, periods=len(dfm[ds]), freq='MS')
        dfm = dfm[sDay_cali:eDay_cali]
        dff = pd.concat([dff, dfm[ds]], axis=1)
    
    # Relative amount change divided by orginal data multiplying 100
    dff = (dff.sub(dff['org'], axis=0).div(dff['org'], axis=0)) * 100
    
    # Delete org column
    del dff['org']

    ds2 = ["d025", "d05", "d075", "d125", "d15", "d175"]
    colors = (
        'brown', 'chocolate', 'gold',
        'greenyellow', 'limegreen', 'forestgreen',)

    print(dff)
    order_st = [5, 4, 3, 0, 1, 2]
    print(order_st)
    fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15, wspace=0.05, hspace=0.05)
    # '''


    # ax.plot(dff)
    for i in order_st:
        ax.fill_between(dff.index, dff[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        ax.plot(dff.index, dff[ds2[i]], c=colors[i], lw=0.5) # surq    
    xlabels = ['D', 'J\n1999', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D',
               'J\n2000', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D' ]
    ax.set_xticks(dff.index) # choose which x locations to have ticks
    ax.set_xticklabels(xlabels) # set the labels to display at those ticks
    ax.set_ylabel('Relative change (%)')
    # ax.xaxis.tick_top()
    
    # Legend
    names = (
        '$D_{0.25}$', '$D_{0.5}$', '$D_{0.75}$',
        '$D_{1.25}$', '$D_{1.5}$', '$D_{1.75}$')


    ps = []
    for c in colors:
        ps.append(Rectangle((0, 0), 0.1, 0.1, fc=c, edgecolor=c, alpha=0.5))
    legend = ax.legend(
        ps, names,
        loc='upper right',
        # title="EXPLANATION",
        handlelength=1., handleheight=1.,        
        edgecolor='none',
        fontsize=10,
        handletextpad=0.5,
        # labelspacing=1,
        columnspacing=1,
        # bbox_to_anchor=(0.5, -0.1),
        ncol=1)
    legend._legend_box.align = "left"
    # plt.legend(handlelength=1, handleheight=1)
    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig('st_sc_drought.png', dpi=300)  
    plt.show()
    # '''


def wt_d():
    sDay = "1/1/1980"
    scsday = "12/1/1998"
    sceday = "12/1/2000"
    wt = "swatmf_out_MF_obs"

    # Open file for loop
    dff_co1l = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        wt = "swatmf_out_MF_obs"

        output = pd.read_csv(
                            os.path.join(wd, wt),
                            delim_whitespace=True,
                            skiprows=1,
                            usecols=[0,],
                            names=[ds]
                            )
        output.index = pd.date_range('1/1/1980', periods=len(output[ds]))
        output = output.resample('MS').mean()
        dfd = output[scsday:sceday]
        dff_co1l = pd.concat([dff_co1l, dfd[ds]], axis=1)

    # dff_co1l = dff_co1l.sub(dff_co1l['org'], axis=0) / (dff_co1l['d175'] - dff_co1l['org']).max()
    
    dff_co1l = (dff_co1l.sub(dff_co1l['org'], axis=0).div(dff_co1l['org'], axis=0)) * 100

    # dff_co1l = (dff_co1l.sub(dff_co1l['org'], axis=0))

    print(dff_co1l)

    # Delete org column
    del dff_co1l['org']

    ds2 = ["d025", "d05", "d075", "d125", "d15", "d175"]
    colors = (
        'brown', 'chocolate', 'gold',
        'greenyellow', 'limegreen', 'forestgreen',)

    # print(dff_co1l)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, 
                            sharey=True
                            )
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15, wspace=0.05, hspace=0.05)
    ax1.set_ylabel('Relative change (%)')
    order_co1l = [5, 4, 3, 0, 1, 2]
    for i in order_co1l:
        axes[0].fill_between(dff_co1l.index, dff_co1l[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[0].plot(dff_co1l.index, dff_co1l[ds2[i]], c=colors[i], lw=0.5) # surq    
    xlabels = ['D', '1999\nJ', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D',
               '2000\nJ', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D' ]
    axes[0].set_xticks(dff_co1l.index) # choose which x locations to have ticks
    axes[0].set_xticklabels(xlabels) # set the labels to display at those ticks
    # print(dff_co1l)
    # dff1 = surqdf[sday:eday].sub(surqdf['org'][sday:eday], axis=0) / (surqdf['w_508'] - surqdf['org']).max()

    # Open file for loop 2nd
    dff_co2d = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        wt = "swatmf_out_MF_obs"

        output = pd.read_csv(
                            os.path.join(wd, wt),
                            delim_whitespace=True,
                            skiprows=1,
                            usecols=[1,],
                            names=[ds]
                            )
        output.index = pd.date_range('1/1/1980', periods=len(output[ds]))
        output = output.resample('MS').mean()
        dfd = output[scsday:sceday]
        dff_co2d = pd.concat([dff_co2d, dfd[ds]], axis=1)
    # dff_co2d = dff_co2d.sub(dff_co2d['org'], axis=0) / (dff_co2d['d175'] - dff_co2d['org']).max()
    dff_co2d = dff_co2d.sub(dff_co2d['org'], axis=0).div(dff_co2d['org'], axis=0) * 100
    
    # Delete org column
    del dff_co2d['org']

    order_co2d = [5, 4, 3, 0, 1, 2]
    for i in order_co2d:
        axes[1].fill_between(dff_co2d.index, dff_co2d[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[1].plot(dff_co2d.index, dff_co2d[ds2[i]], c=colors[i], lw=0.5) # surq    
    axes[1].set_xticks(dff_co2d.index) # choose which x locations to have ticks
    axes[1].set_xticklabels(xlabels) # set the labels to display at those ticks
    axes[0].xaxis.tick_top()
    axes[1].xaxis.tick_top()
    # axes[1].yaxis.tick_right()
    
    # Legend
    names = (
        '$D_{0.25}$', '$D_{0.5}$', '$D_{0.75}$',
        '$D_{1.25}$', '$D_{1.5}$', '$D_{1.75}$')

    ps = []
    for c in colors:
        ps.append(Rectangle((0, 0), 0.1, 0.1, fc=c, edgecolor=c, alpha=0.5))
    legend = ax1.legend(
        ps, names,
        loc='lower center',
        # title="EXPLANATION",
        handlelength=1., handleheight=1.,        
        edgecolor='none',
        fontsize=10,
        handletextpad=0.5,
        # labelspacing=1,
        columnspacing=1,
        bbox_to_anchor=(0.5, -0.2),
        ncol=8)
    legend._legend_box.align = "left"
    # plt.legend(handlelength=1, handleheight=1)
    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)
    
    axes[0].text(0.03, 0.9, "a) CO1L", transform=axes[0].transAxes)
    axes[1].text(0.03, 0.9, "b) CO2D", transform=axes[1].transAxes)

    plt.savefig('wt_sc_drought.png', dpi=300)  
    plt.show()

   
def data_analysis():
    pcp = "pcp1.pcp"
    tmp = "tmp1.tmp"

    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    with open(os.path.join(mwd, pcp), "r") as f:
        data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
    # date = [x.strip().split() for x in data if x.strip().startswith("month:")]  # Collect only lines with dates
    # yr = [x[0:4] for x in data] # Only date
    pdata = [float(x[7:12]) for x in data] # Only date
    df = pd.DataFrame({'ppt':pdata})

    df.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    df.replace(-99.0, np.nan, inplace=True)

    # Get simulated year
    df_s = df['1993-1-1':'2012-12-31']

    # Get drought year
    df_d = df['1999-1-1':'1999-12-31']

    # Flood year
    df_f = df['2004-1-1':'2004-12-31']

    dfm_s = df_s.resample('MS').sum()
    dfa_s = df_s.resample('AS').sum()

    # print(dfm_s)
    print(dfm_s['1999-1-1':'1999-12-1'])

    '''
    dfm_s_f = dfm_s['ppt'].values
    dfm_s_color = plt.cm.Spectral(dfm_s_f/dfm_s_f.max())
    dfa_s_f = dfa_s['ppt'].values
    dfa_s_color = plt.cm.gist_rainbow(dfa_s_f/dfa_s_f.max())


    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(dfm_s.index, dfm_s.ppt, 20, color=dfm_s_color, zorder=4, alpha = 1) 

    ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.bar(dfa_s.index, dfa_s.ppt, 350, color=dfa_s_color, alpha=0.3, align='edge',zorder=1)
    
    ax2.bar(dfa_s.index[6], 600, 350, edgecolor="k", lw=2, align='edge', fill=False) 
    ax2.bar(dfa_s.index[11], 1560, 350, edgecolor="k", lw=2, align='edge', fill=False)

    # Annotation
    ax1.annotate(
        'Drought', xy=(0.33, 0.35), xycoords=ax1.transAxes,
        xytext=(0.33, 1.05), textcoords=ax1.transAxes,
        horizontalalignment='center',
        arrowprops=dict(arrowstyle='<|-'))

    ax1.annotate(
        'Flood', xy=(0.575, 0.88), xycoords=ax1.transAxes,
        xytext=(0.575, 1.05), textcoords=ax1.transAxes,
        horizontalalignment='center',
        arrowprops=dict(arrowstyle='<|-'))

    print(dfa_s.max())

    ax1.set_xticks(dfa_s.index) # choose which x locations to have ticks
    xlabels = [x for x in range(1993, 2013)]
    ax1.set_xticklabels(xlabels) # set the labels to display at those ticks
  
    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax1.margins(x=0.01)
    ax1.set_ylabel('Monthly total precipitation [mm]', labelpad=30)
    ax2.set_ylabel('Annual total precipitation [mm]', labelpad=30)    
    ax1.set_ylim(0, 800)
    ax2.set_ylim(0, 1800)    
    ax2.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax3 = fig.add_subplot(111, frameon=False)
    ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    # # get bounding box information for the axes (since they're in a line, you only care about the top and bottom)
    # bbox_ax = ax3.get_position()

    ax4 = fig.add_subplot(111, frameon=False)
    ax4.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    ax3 = plt.imshow(np.array([[dfm_s_f.min(), dfm_s_f.max()]]), cmap=plt.cm.Spectral)
    ax3.set_visible(False)
    ax4 = plt.imshow(np.array([[0, dfa_s_f.max()]]), cmap=plt.cm.gist_rainbow)
    ax4.set_visible(False)



    cbaxes2 = fig.add_axes([0.94, 0.11, 0.01, 0.67])
    cb1 = plt.colorbar(
                    ax4,
                    cax = cbaxes2, orientation="vertical",
                    # ticks = [0,  1500]
                    )
    cb1.outline.set_visible(False)
    # cb1.set_ticks([])
    # cb1.ax.set_yticklabels(['', ''])
    cb1.ax.tick_params(labelsize= 7)
    cb1.set_alpha(0.2)
    cb1.draw_all()
    cb1.ax.yaxis.set_ticks_position('left')
    cb1.ax.yaxis.set_label_position('left')
    cb1.ax.set_yticklabels(['','','','',])

    cbaxes1 = fig.add_axes([0.08, 0.11, 0.01, 0.364])
    cb = plt.colorbar(
                    ax3,
                    cax = cbaxes1, orientation="vertical",
                    ticks = [0,  100, 200, 300]
                    )
    cb.outline.set_visible(False)
    # cb1.set_ticks([])
    cb.ax.set_yticklabels(['','','','',])
    cb.ax.tick_params(labelsize= 7)
    cb.set_alpha(1)
    cb.draw_all()
    cb.ax.set_yticklabels(['','','','',])
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    plt.savefig('pcp_scenario_I.png', dpi=300)  
    plt.show()
    '''

def create_pcp():
    pcp = "pcp1.pcp"
    tmp = "tmp1.tmp"

    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    with open(os.path.join(wd, pcp), "r") as f:
        data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
    # date = [x.strip().split() for x in data if x.strip().startswith("month:")]  # Collect only lines with dates
    # yr = [x[0:4] for x in data] # Only date
    pdata = [float(x[7:12]) for x in data] # Only date
    df = pd.DataFrame({'ppt':pdata})

    df.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    df.replace(-99.0, np.nan, inplace=True)

    # Get simulated year
    df_s = df['1993-1-1':'2012-12-31']

    # Get drought year
    df_d = df['1999-1-1':'1999-12-31']
    dfm_d = df_d.resample('MS').sum()

    # Flood year
    df_f = df['2004-1-1':'2004-12-31']
    dfm_f = df_f.resample('MS').sum()


    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    # ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    
    ax1 = fig.add_subplot(121, frameon=False)
    ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax2 = fig.add_subplot(122, frameon=False)
    ax2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # df_d_f = df_d['ppt'].values
    # df_d_color = plt.cm.gist_rainbow_r(df_d_f/df_d_f.max())
    # ax1.bar(df_d.index, df_d.ppt, 20, color=df_d_color, alpha=0.5, align='edge')   

    dfm_d_f = dfm_d['ppt'].values
    dfm_d_color = plt.cm.autumn(dfm_d_f/dfm_d_f.max())
    axes[0].bar(dfm_d.index, dfm_d.ppt, 30, color=dfm_d_color, alpha=0.5, align='center', zorder=0, edgecolor='w')   
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.25, marker='v', c='w', edgecolors='k', label='$D_{0.25}$') # 
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.5, marker='<', c='w', edgecolors='k', label=r'$D_{0.5}$')  
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.75, marker='>', c='w', edgecolors='k', label=r'$D_{0.75}$')
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.25, marker='s', c='w', edgecolors='k', label=r'$D_{1.25}$') # 
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.5, marker='o', c='w', edgecolors='k', label=r'$D_{1.5}$')  
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.75, marker='^', c='w', edgecolors='k', label=r'$D_{1.75}$')

    axes[0].set_xticks(dfm_d.index) # choose which x locations to have ticks
    axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ]) # set the labels to display at those ticks
    axes[0].set_ylabel(r'Monthly precipitation for drought year $[mm]$', fontsize=10, labelpad=25)
    ax1 = plt.imshow(np.array([[dfm_d_f.min(), dfm_d_f.max()]]), cmap=plt.cm.autumn)
    ax1.set_visible(False)
    # cbaxes = fig.add_axes([0.015, 0.14, 0.02, 0.41])
    # cb = plt.colorbar(ax1,
    #                 cax = cbaxes, orientation="vertical",
    #                 ticks = [0,  50, 100]
    #                 )
    # cb.outline.set_visible(False)
    # cb.set_alpha(0.3)
    # cb.draw_all()
    # # cb1.set_ticks([])
    # cb.ax.set_yticklabels(['', ''])

    # reverse the order
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1])


    # Food
    dfm_f_f = dfm_f['ppt'].values
    dfm_f_color = plt.cm.cool(dfm_f_f/dfm_f_f.max())
    axes[1].bar(dfm_f.index, dfm_f.ppt, 30, color=dfm_f_color, alpha=0.5, align='center', zorder=0, edgecolor='w')   
    axes[1].scatter(dfm_f.index, dfm_f.ppt*0.25, marker='v', c='w', edgecolors='k', label='$F_{0.25}$') # 
    axes[1].scatter(dfm_f.index, dfm_f.ppt*0.5, marker='<', c='w', edgecolors='k', label=r'$F_{0.5}$')  
    axes[1].scatter(dfm_f.index, dfm_f.ppt*0.75, marker='>', c='w', edgecolors='k', label=r'$F_{0.75}$')
    axes[1].scatter(dfm_f.index, dfm_f.ppt*1.25, marker='s', c='w', edgecolors='k', label=r'$F_{1.25}$') # 
    axes[1].scatter(dfm_f.index, dfm_f.ppt*1.5, marker='o', c='w', edgecolors='k', label=r'$F_{1.5}$')  
    axes[1].scatter(dfm_f.index, dfm_f.ppt*1.75, marker='^', c='w', edgecolors='k', label=r'$F_{1.75}$')

    axes[1].set_xticks(dfm_f.index) # choose which x locations to have ticks
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ]) # set the labels to display at those ticks
    axes[1].set_ylabel(r'Monthly precipitation for flooding year $[mm]$', fontsize=10, labelpad=25)
    ax2 = plt.imshow(np.array([[dfm_f_f.min(), dfm_f_f.max()]]), cmap=plt.cm.cool)
    ax2.set_visible(False)
    

    cbaxes1 = fig.add_axes([0.525, 0.145, 0.015, 0.405])
    cb1 = plt.colorbar(ax2,
                    cax = cbaxes1, orientation="vertical",
                    ticks = [0, 100, 200]
                    )
    # cb1.outline.set_visible(False)

    # cb1.set_ticks([])
    cb1.ax.set_yticklabels(['', ''])
    # cb1.set_alpha(0.3)
    # cb1.draw_all()
    # reverse the order
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1])


    cbaxes = fig.add_axes([0.045, 0.14, 0.015, 0.41])
    cb = plt.colorbar(ax1,
                    cax = cbaxes, orientation="vertical",
                    ticks = [0,  50, 100]
                    )
    # cb.outline.set_visible(False)
    # cb.set_alpha(0.3)
    # cb.draw_all()
    # cb1.set_ticks([])
    cb.ax.set_yticklabels(['', ''])


    axes[0].text(0.03, 0.95, "a)", transform=axes[0].transAxes)
    axes[1].text(0.95, 0.95, "b)", transform=axes[1].transAxes)

    plt.savefig('pcp_scenario.png', dpi=300)  
    plt.show()


def create_pcp2():
    pcp = "pcp1.pcp"
    tmp = "tmp1.tmp"
    wd = ("d:\\SG_papers\\2018_Park_et_al_QuantifyingWaterUsingSM\\scripts")
    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    with open(os.path.join(wd, pcp), "r") as f:
        data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
    # date = [x.strip().split() for x in data if x.strip().startswith("month:")]  # Collect only lines with dates
    # yr = [x[0:4] for x in data] # Only date
    pdata = [float(x[7:12]) for x in data] # Only date
    df = pd.DataFrame({'ppt':pdata})

    df.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    df.replace(-99.0, np.nan, inplace=True)

    # Get simulated year
    df_s = df['1993-1-1':'2012-12-31']

    # Get drought year
    df_d = df['1999-1-1':'1999-12-31']
    dfm_d = df_d.resample('MS').sum()

    # Flood year
    df_f = df['2004-1-1':'2004-12-31']
    dfm_f = df_f.resample('MS').sum()


    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    # ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    
    ax1 = fig.add_subplot(121, frameon=False)
    ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax2 = fig.add_subplot(122, frameon=False)
    ax2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # df_d_f = df_d['ppt'].values
    # df_d_color = plt.cm.gist_rainbow_r(df_d_f/df_d_f.max())
    # ax1.bar(df_d.index, df_d.ppt, 20, color=df_d_color, alpha=0.5, align='edge')   

    dfm_d_f = dfm_d['ppt'].values
    dfm_d_color = plt.cm.autumn(dfm_d_f/dfm_d_f.max())
    axes[0].bar(dfm_d.index, dfm_d.ppt, 30, color=dfm_d_color, alpha=0.5, align='center', zorder=0, edgecolor='w')   
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.25, marker='v', c='w', edgecolors='k', label='$D_{0.25}$') # 
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.5, marker='<', c='w', edgecolors='k', label=r'$D_{0.5}$')  
    axes[0].scatter(dfm_d.index, dfm_d.ppt*0.75, marker='>', c='w', edgecolors='k', label=r'$D_{0.75}$')
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.25, marker='s', c='w', edgecolors='k', label=r'$D_{1.25}$') # 
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.5, marker='o', c='w', edgecolors='k', label=r'$D_{1.5}$')  
    axes[0].scatter(dfm_d.index, dfm_d.ppt*1.75, marker='^', c='w', edgecolors='k', label=r'$D_{1.75}$')

    axes[0].set_xticks(dfm_d.index) # choose which x locations to have ticks
    axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ]) # set the labels to display at those ticks
    axes[0].set_ylabel(r'Monthly precipitation for drought year $[mm]$', fontsize=10, labelpad=25)
    ax1 = plt.imshow(np.array([[dfm_d_f.min(), dfm_d_f.max()]]), cmap=plt.cm.autumn)
    ax1.set_visible(False)

    cbaxes = fig.add_axes([0.045, 0.14, 0.015, 0.41])
    cb = plt.colorbar(ax1,
                    cax = cbaxes, orientation="vertical",
                    ticks = [0,  50, 100]
                    )
    # cb.outline.set_visible(False)
    # cb.set_alpha(0.3)
    # cb.draw_all()
    # cb1.set_ticks([])

    cb.set_alpha(0.5)
    cb.draw_all()
    cb.ax.set_yticklabels(['', ''])

    axes[0].text(0.03, 0.95, "a)", transform=axes[0].transAxes)
    axes[1].text(0.95, 0.95, "b)", transform=axes[1].transAxes)

    # reverse the order
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1])

    # Food
    pcp = "pcp1.pcp"
    wds = [d_127, d_254, d_381, d_508, w_127, w_254, w_381, w_508, org]
    datasets = ["d_127", "d_254", "d_381", "d_508", "w_127", "w_254", "w_381", "w_508", "org"]

    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    pcpdf = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        with open(os.path.join(wd, pcp), "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        pdata = [float(x[7:12]) for x in data] # Only data

        add_pcpdf = pd.DataFrame({ds: pdata})
        pcpdf = pd.concat([pcpdf, add_pcpdf[ds]], axis=1)
    pcpdf.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    dff = pcpdf['2004-7-27':'2004-8-1']
    wdff = pcpdf['2004-7-28':'2004-7-30']
    # print(wdff)
    # dff.replace(-99.0, np.nan, inplace=True)
    # s1 = pd.Series(pd.factorize(dff.index)[0] + 1, dff.index)
    # x_smooth = np.linspace(s1.min(), s1.max(), 100) #300 represents number of points to make between T.min and T.max
    # y_smooth = spline(s1.values, dff.d_508, x_smooth)
    # plt.plot(x_smooth, y_smooth)
    axes[1].bar(dff.index, (dff["d_508"]/dff["d_508"].max()*-1), color='indigo')
    axes[1].bar(dff.index, (dff["d_381"]/dff["d_508"].max()*-1), color='darkviolet')    
    axes[1].bar(dff.index, (dff["d_254"]/dff["d_508"].max()*-1), color='mediumorchid')   
    axes[1].bar(dff.index, (dff["d_127"]/dff["d_508"].max()*-1), color='violet')
    axes[1].bar(dff.index, (dff["w_508"]/dff["w_508"].max()), color='indigo')
    axes[1].bar(dff.index, (dff["w_381"]/dff["w_508"].max()), color='darkviolet')    
    axes[1].bar(dff.index, (dff["w_254"]/dff["w_508"].max()), color='mediumorchid')   
    axes[1].bar(dff.index, (dff["w_127"]/dff["w_508"].max()), color='violet')
    barlist = axes[1].bar(wdff.index, (wdff["w_127"]/dff["w_508"].max()), color='royalblue')
    barlist[0].set_color('skyblue')
    barlist[2].set_color('lightskyblue')    

    # moving bottom spine up to y=0 position:
    axes[1].xaxis.set_ticks_position('bottom')
    axes[1].spines['bottom'].set_position(('data',0))
    xlims = axes[1].get_xlim()
    axes[1].set_xticks(xlims)
    # axes[1].axis('off')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(top='off', bottom='on', left='off', right='off')
    
    # Turn off tick labels
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels(["Jul\n2004", "Aug\n2004"])

    # Legend
    names = (
        '$P_{20}$', '$P_{15}$', '$P_{10}$', '$P_{5}$',
        '$P_{2.4}$', '$P_{0.5 - 0.6}$')
    colors = (
        'indigo', 'darkviolet', 'mediumorchid', 'violet', 'royalblue', 'lightskyblue')
    ps = []
    for c in colors:
        ps.append(Rectangle((0, 0), width=0.1, height=0.125, fc=c, alpha=1))
    legend = axes[1].legend(
        ps, names,
        loc='upper left',
        title="EXPLANATION",
        edgecolor='none',
        fontsize=10,
        bbox_to_anchor=(-0.05, 1),
        ncol=1)
    legend._legend_box.align = "left"

    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)

    # # Annotation
    # axes[1].annotate(
    #     '', xy=(0.2, 0.65), xycoords=axes[1].transAxes,
    #     xytext=(0.2, .35), textcoords=axes[1].transAxes,
    #     horizontalalignment='center',
    #     arrowprops=dict(color='grey', arrowstyle='-'))

    # axes[1].annotate(
    #     'Flood', xy=(0.575, 0.88), xycoords=axes[1].transAxes,
    #     xytext=(0.575, 1.05), textcoords=axes[1].transAxes,
    #     horizontalalignment='center',
    #     arrowprops=dict(arrowstyle='<|-'))

    axes[1].text(.39, .6, 'Wet', style='italic', transform=axes[1].transAxes)
    axes[1].text(.39, .4, 'Dry', style='italic', transform=axes[1].transAxes)

    plt.savefig('pcp_scenario222.png', dpi=300)  
    plt.show()


def drought_pcp_file():
    pcp = "pcp1.pcp"
    tmp = "tmp1.tmp"

    # read precipitation data
    y = ("Station", "Lati", "Long", "Elev") # Remove unnecssary lines
    with open(os.path.join(wd, pcp), "r") as f:
        data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
    # date = [x.strip().split() for x in data if x.strip().startswith("month:")]  # Collect only lines with dates
    # yr = [x[0:4] for x in data] # Only date
    pdata = [float(x[7:12]) for x in data] # Only date
    df = pd.DataFrame({'ppt':pdata})

    df.index = pd.date_range(startDate, periods=len(pdata), freq='D')
    df.replace(-99.0, np.nan, inplace=True)

    # Make a dataframe for all of the scenarios
    scnames = ['d025', 'd05', 'd075', 'd125', 'd15', 'd175']
    scfactors = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
 
    df['date'] = df.index
    df['year'] = df['date'].dt.year
    df['doy'] = df['date'].dt.dayofyear
    df['doy'] = df['doy'].apply('{:03d}'.format)

    for sn, sf in zip(scnames, scfactors):
        df[sn] = df.ppt['1999-1-1':'1999-12-31']*sf
        df[sn] = df[sn].fillna(df['ppt'])
        # df[sn] = df[sn].map(lambda n: '{:05.1f}'.format(n))
      
    fig, ax = plt.subplots()
    for sn in scnames:
        ax.plot(df[sn]['1999-1-1':'1999-12-31'])
    # ax.plot(df['d025']['1999-1-1':'1999-12-31'])
    # ax.plot(df['d175']['1999-1-1':'1999-12-31'])
    plt.show()
    df.fillna(-99.0, inplace=True)
    for sn, sf in zip(scnames, scfactors):
        df[sn] = df[sn].apply(lambda n: '{:05.1f}'.format(n) if n != -99.0 else '{:.1f}'.format(n))
        df[sn] = df.year.map(str)+df.doy.map(str)+df[sn]

    for sn in scnames:    
        with open(os.path.join(wd, "pcp1.pcp." + sn), 'w') as f:
                f.write("Station  MBRW_PCP,"+ "\n" + 
                        "Lati    31.4" + "\n" + 
                        "Long   -97.4" + "\n" +
                        "Elev     220" + "\n")
                df[sn].to_csv(f, header=False, index=False)


def w_bal():
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mtick

    # Open file for loop
    surqdf = pd.DataFrame()
    latqdf = pd.DataFrame()
    gwqdf = pd.DataFrame()
    percodf = pd.DataFrame()
    swgwdf = pd.DataFrame()
    swdf = pd.DataFrame()
    gwdf = pd.DataFrame()
    for wd, ds in zip(wds, datasets):
        with open(os.path.join(wd, "output.std"), "r") as infile:
                lines = []
                y = ("TIME", "UNIT", "SWAT", "(mm)")
                for line in infile:
                    data = line.strip()
                    if len(data) > 100 and not data.startswith(y):  # 1st filter
                        lines.append(line)
        eYear = '2012'
        dates = []
        for line in lines:  # 2nd filter
            try:
                date = line.split()[0]
                if (date == eYear):  # Stop looping
                    break
                elif(len(str(date)) == 4):  # filter years
                    continue
                else:
                    dates.append(line)
            except:
                pass
        date_f, prec, surq, latq, gwq, swgw, perco, tile, sw, gw = [], [], [], [], [], [], [], [], [], []
        for i in range(len(dates)):
            # date_f.append(int(dates[i].split()[0]))
            # prec.append(float(dates[i].split()[1]))
            surq.append(float(dates[i].split()[2]))
            latq.append(float(dates[i].split()[3]))
            gwq.append(float(dates[i].split()[4]))
            swgw.append(float(dates[i].split()[5]))
            perco.append(float(dates[i].split()[6]))
            # tile.append(float(dates[i].split()[7]))  # not use it for now
            sw.append(float(dates[i].split()[9]))
            gw.append(float(dates[i].split()[10]))

        add_surqdf = pd.DataFrame({ds: surq})
        surqdf = pd.concat([surqdf, add_surqdf[ds]], axis=1)
        add_latqdf = pd.DataFrame({ds: latq})
        latqdf = pd.concat([latqdf, add_latqdf[ds]], axis=1)
        add_gwqdf = pd.DataFrame({ds: gwq})
        gwqdf = pd.concat([gwqdf, add_gwqdf[ds]], axis=1)
        add_swgwdf = pd.DataFrame({ds: swgw})
        swgwdf = pd.concat([swgwdf, add_swgwdf[ds]], axis=1)
        add_percodf = pd.DataFrame({ds: perco})
        percodf = pd.concat([percodf, add_percodf[ds]], axis=1)
        add_swdf = pd.DataFrame({ds: sw})
        swdf = pd.concat([swdf, add_swdf[ds]], axis=1)
        add_gwdf = pd.DataFrame({ds: gw})
        gwdf = pd.concat([gwdf, add_gwdf[ds]], axis=1)

    surqdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    latqdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    gwqdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    swgwdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    percodf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    swdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')
    gwdf.index = pd.date_range('1/1/1985', periods=len(surqdf), freq='MS')

    sday = "12/1/1998"
    eday = "12/1/2000"
    # dff1 = surqdf[['d_127', 'd_254']]['7/1/2004':'11/1/2004']
    dff1 = surqdf[sday:eday]
    dff2 = latqdf[sday:eday]
    dff3 = gwqdf[sday:eday]
    dff4 = swgwdf[sday:eday]
    dff5 = percodf[sday:eday]
    dff6 = swdf[sday:eday]
    dff7 = gwdf[sday:eday]

    # # ===== Relative change (Maximum to 1 divide by maximum value in the dataset)
    # dff1 = (surqdf[sday:eday].sub(surqdf['org'][sday:eday], axis=0)).div(surqdf['org'][sday:eday], axis=0) * 100
    # dff2 = latqdf[sday:eday].sub(latqdf['org'][sday:eday], axis=0) / (latqdf['d175'] - latqdf['org']).max()
    # dff3 = gwqdf[sday:eday].sub(gwqdf['org'][sday:eday], axis=0) / (gwqdf['d175'] - gwqdf['org']).max()
    # dff4 = swgwdf[sday:eday].sub(swgwdf['org'][sday:eday], axis=0) / (swgwdf['d175'] - swgwdf['org']).max()
    # dff5 = percodf[sday:eday].sub(percodf['org'][sday:eday], axis=0) / (percodf['d175'] - percodf['org']).max()
    # dff6 = swdf[sday:eday].sub(swdf['org'][sday:eday], axis=0) / abs((swdf['d025'] - swdf['org']).min())    
    # dff7 = gwdf[sday:eday].sub(gwdf['org'][sday:eday], axis=0) / abs((gwdf['d175'] - gwdf['org']).min())

    # ===  Relative amount change (Maximum can be different; divide by orginal)
    dff1 = (dff1.sub(dff1['org'], axis=0).div(dff1['org'], axis=0)) * 100
    dff2 = (dff2.sub(dff2['org'], axis=0).div(dff2['org'], axis=0)) * 100
    dff3 = (dff3.sub(dff3['org'], axis=0).div(dff3['org'], axis=0)) * 100
    dff4 = (dff4.sub(dff4['org'], axis=0).div(dff4['org'], axis=0)) * 100
    dff5 = (dff5.sub(dff5['org'], axis=0).div(dff5['org'], axis=0)) * 100
    dff6 = (dff6.sub(dff6['org'], axis=0).div(dff6['org'], axis=0)) * 100
    dff7 = (dff7.sub(dff7['org'], axis=0).div(dff7['org'], axis=0)) * 100


    dff4 = dff4.replace([np.inf, -np.inf], np.nan)
    dff5 = dff5.replace([np.inf, -np.inf], np.nan)
    dff1 = dff1.fillna(0)
    dff4 = dff4.fillna(0)
    dff5 = dff5.fillna(0)

    # Delete org column
    del dff1['org'], dff2['org'], dff3['org'], dff4['org'], dff5['org'], dff6['org'], dff7['org']

    fig, axes = plt.subplots(4, 2, figsize=(8, 8))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
    
    ax1.set_ylabel('Relative change (%)', labelpad=15)
    order_surq = [5, 4, 3, 0, 1, 2]
    for i in order_surq:
        axes[0, 0].fill_between(dff1.index, dff1[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[0, 0].plot(dff1.index, dff1[ds2[i]], c=colors[i], lw=0.5) # surq

    order_latq = [5, 4, 3, 0, 1, 2]
    for i in order_latq:
        axes[0, 1].fill_between(dff2.index, dff2[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[0, 1].plot(dff2.index, dff2[ds2[i]], c=colors[i], lw=0.5) # surq

    order_sw = [5, 4, 3, 0, 1, 2]
    for i in order_sw:
        axes[1, 0].fill_between(dff6.index, dff6[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[1, 0].plot(dff6.index, dff6[ds2[i]], c=colors[i], lw=0.5) # surq

    order_gwq = [5, 4, 3, 0, 1, 2]
    for i in order_gwq:
        axes[1, 1].fill_between(dff3.index, dff3[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[1, 1].plot(dff3.index, dff3[ds2[i]], c=colors[i], lw=0.5) # surq

    order_swgw = [5, 4, 3, 0, 1, 2]
    for i in order_swgw:
        axes[2, 0].fill_between(dff4.index, dff4[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[2, 0].plot(dff4.index, dff4[ds2[i]], c=colors[i], lw=0.5) # surq
    
    order_perco = [5, 4, 3, 0, 1, 2]
    for i in order_perco:
        axes[2, 1].fill_between(dff5.index, dff5[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[2, 1].plot(dff5.index, dff5[ds2[i]], c=colors[i], lw=0.5) # surq        
    order_gw = [5, 4, 3, 0, 1, 2]
    for i in order_gw:
        axes[3, 0].fill_between(dff7.index, dff7[ds2[i]], y2=0, facecolor=colors[i], alpha=0.5) # surq
        axes[3, 0].plot(dff7.index, dff7[ds2[i]], c=colors[i], lw=0.5) # surq        

    axes[0, 0].set_title('SURQ', fontsize=10, x=0.08, y=.85)
    axes[0, 1].set_title('LATQ', fontsize=10, x=0.08, y=.85)
    axes[1, 0].set_title('SW', fontsize=10, x=0.08, y=.85)
    axes[1, 1].set_title('GWQ', fontsize=10, x=0.08, y=.85)
    axes[2, 0].set_title('SWGW', fontsize=10, x=0.08, y=.85)
    axes[2, 1].set_title('PERCO', fontsize=10, x=0.08, y=.85)
    axes[3, 0].set_title('GW', fontsize=10, x=0.08, y=.85)

    for ax in axes.flat:
        # ax.set_ylim(-1.1, 1.1)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.tick_params(axis='both', labelsize=8)

    axes[1, 0].tick_params(labelcolor='k', left='on', bottom='off')
    axes[1, 0].set_xticks([])

    # axes[2, 0].set_ylim(-1.1, 0.3)
    # axes[0, 0].tick_params(labelcolor='k', left='on', bottom='off')
    # axes[1, 0].tick_params(labelcolor='k', left='on', bottom='off')
    axes[2, 0].tick_params(labelcolor='k', left='on', bottom='off')
    axes[2, 0].set_xticks([])
    # axes[1, 1].tick_params(labelcolor='k', bottom='on', left='off')
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].tick_params(labelcolor='k', left='on')
    axes[0, 0].set_xticks(dff1.index) # choose which x locations to have ticks
    xlabels = ['D', '1999\nJ', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D',
               '2000\nJ', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D' ]
    axes[0, 0].set_xticklabels(xlabels) # set the labels to display at those ticks

    axes[0, 1].xaxis.tick_top()
    axes[0, 1].tick_params(labelcolor='k', right='on')
    axes[0, 1].set_xticks(dff1.index) # choose which x locations to have ticks
    # xlabels = ['2004\nM', 'J', 'J', 'A', 'S', 'O', 'N', 'D', '2005\nJ', 'F', 'M', 'A']
    axes[0, 1].set_xticklabels(xlabels) # set the labels to display at those ticks 
    # axes[0, 1].set_yticks([])
    axes[0, 1].tick_params(labelcolor='k', right='on')
    axes[1, 1].tick_params(labelcolor='k', right='on', bottom='off')
    axes[1, 1].set_xticks([])
    axes[2, 1].tick_params(labelcolor='k', right='on', top='off')        
    axes[0, 1].yaxis.tick_right()
    axes[1, 1].yaxis.tick_right()
    axes[2, 1].yaxis.tick_right()

    xlabels2 = ['D', 'J\n1999', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D',
               'J\n2000', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D' ]
    # axes[3, 0].set_ylim(-1.1, 0.3)
    # axes[0, 0].tick_params(labelcolor='k', left='on', bottom='off')
    # axes[1, 0].tick_params(labelcolor='k', left='on', bottom='off')
    axes[3, 0].tick_params(labelcolor='k', left='on', bottom='on')
    axes[3, 0].set_xticks(dff1.index)
    axes[3, 0].set_xticklabels(xlabels2)
    axes[3, 1].axis('off')
    # axes[2, 1].axis('off')
    axes[2, 1].tick_params(labelcolor='k', bottom='on', top='off')
    axes[2, 1].set_xticks(dff1.index)
    axes[2, 1].set_xticklabels(xlabels2)

    # axes[0, 0].ticklabel_format(
    #     style='sci', axis='y', scilimits=(0, 0))    

    axes[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axes[2, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # Legend
    names = (
        '$D_{0.25}$', '$D_{0.5}$', '$D_{0.75}$',
        '$D_{1.25}$', '$D_{1.5}$', '$D_{1.75}$')

    ps = []
    for c in colors:
        ps.append(Rectangle((0, 0), 0.1, 0.1, fc=c, edgecolor=c, alpha=0.5))
    legend = ax1.legend(
        ps, names,
        loc='lower right',
        title="EXPLANATION",
        handlelength=1., handleheight=1.,        
        edgecolor='none',
        fontsize=8,
        handletextpad=0.5,
        # labelspacing=1,
        columnspacing=1,
        bbox_to_anchor=(1.02, 0.13),
        ncol=8)
    legend._legend_box.align = "left"
    # plt.legend(handlelength=1, handleheight=1)
    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=10)

    ax1.text(0.52, -0.01, 
                "SURQ:\n"+
                "LATQ:\n" +
                "SW:\n"+
                "GWQ:\n"+
                "SWGW:\n"+
                "PERCO:\n"+
                "GW:",
                fontsize=9)
    ax1.text(0.6, -0.01, 
                "Surface runoff to streams\n"+
                "Lateral flow to streams\n" +
                "Total soil water contained in the watershed\n"+
                "Groundwater flow to streams\n"+
                "Seepage from streams to the aquifer\n"+
                "Deep percolation (recharge) to groundwater\n"+
                "Total groundwater contained in the watershed",
                fontsize=9)

    plt.savefig('w_bal_drought.png', dpi=150)      
    plt.show()


# streamflow_month()
def sg_int():
    import scipy.stats as ss
    import operator
    import shapefile
    from matplotlib import style
    from matplotlib import patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    
    filename = "swatmf_out_MF_gwsw_monthly"
    gwsw_a = "swatmf_out_MF_gwsw_yearly"
    # y = ("for", "Positive:", "Negative:", "Daily", "Monthly", "Annual", "Layer,", "month") # Remove unnecssary lines

    # with open(os.path.join(wd, filename), "r") as f:
    #     data = [(x.split()[3]) for x in f if x.strip() and not x.strip().startswith(y)]  
    # # date = [x.strip().split() for x in data if x.strip().startswith("month:")]
    # # data1 = [x.split() for x in data]
    # # onlyDate = [x[1] for x in date] 
    # #dateList = [(sdate + datetime.timedelta(months = int(i)-1)).strftime("%m-%Y") for i in onlyDate]
    # dateList = pd.date_range(startDate, periods=396, freq = 'M').strftime("%b-%Y").tolist()
    # # print(len(data)/1467)
    # print(dateList)
    adata = np.loadtxt(
                        os.path.join(wd, gwsw_a),
                        skiprows=5,
                        comments=["year:", "Layer"])
    adf = np.reshape(adata[:, 3], (int(len(adata)/1467), 1467))
    adf2 = pd.DataFrame(adf)
    adf2.index = pd.date_range(startDate, periods=len(adf2), freq='A').strftime("%Y")
    adff = (adf2.mean())

    data = np.loadtxt(
                        os.path.join(wd, filename),
                        skiprows=5,
                        comments=["month:", "Layer"])
    df = np.reshape(data[:, 3], (int(len(data)/1467), 1467))
    df2 = pd.DataFrame(df)
    df2.index = pd.date_range(startDate, periods=396, freq = 'M').strftime("%b-%Y")

    # df3 = df2.groupby(pd.TimeGrouper(freq="M")).sum()

    df2['date'] = pd.to_datetime(df2.index)
    df2['month'] = df2['date'].dt.month
    # df2['mon'] = df2.DatetimeIndex.month
    df3 = df2.groupby('month').aggregate('mean')

    mar = df3.loc[3] - adff
    jun = df3.loc[6] - adff
    sep = df3.loc[9] - adff
    dec = df3.loc[12] - adff

    # gwsw = pd.concat([mar, jun, sep, dec], axis=1)



    geol = shapefile.Reader(os.path.join(wd, "gis\\mf_boundary.shp")) # Subbasin
    sm_riv = shapefile.Reader(os.path.join(wd, "gis\\mf_riv1.shp" )) # mf_riv
    riv1 = shapefile.Reader(os.path.join(wd, "gis\\riv_SM.shp" )) # swat_riv
    colormap_n = plt.cm.autumn(np.linspace(0., 1, 128))
    colormap_p = plt.cm.winter_r(np.linspace(0., 1, 128))
    colormap_a = plt.cm.gist_rainbow_r(np.linspace(0., 1, 128))
    colormap_a2 = plt.cm.gist_rainbow_r

    # ------------------------------------------------------------------------------
    sr = sm_riv.shapes() # property of sm_river
    coords = [sr[i].bbox for i in range(len(sr))] # get coordinates for each river cell
    width = abs(coords[0][2] - coords[0][0]) # get width for bar plot
    nSM_riv = 1467 # Get number of river cells
    x_min, x_max, y_min, y_max = -169236.13, -129126.13, 928605.52, 952645.5

    # mar.append([1])
    # Sort coordinates by row
    c_sorted = sorted(coords, key=operator.itemgetter(0))

    # Put coordinates and gwsw data in Dataframe together
    f_c = pd.DataFrame(c_sorted, columns=['x_min', 'y_min', 'x_max', 'y_max'])
    f_c['mar'] = mar
    f_c['jun'] = jun
    f_c['sep'] = sep
    f_c['dec'] = dec
    f_c['annual'] = adff

    dff = f_c.drop(['x_max', 'y_max'], axis=1)
    print(dff[['mar', 'jun', 'sep', 'dec']].min(axis=0))
    print(dff[['mar', 'jun', 'sep', 'dec']].max(axis=0))
    print(f_c['annual'].min(axis=0))
    print(f_c['annual'].max(axis=0))
    ##########
    verExg = 150
    widthExg = 1
    gwsw_max = 67 
    gwsw_min = -65

    mar_f = f_c['mar'].values
    print(type(mar_f))
    # arr[arr > 255] = x
    
    mar_p = np.where(mar_f < 0., 0., mar_f)
    mar_colors_p = plt.cm.autumn_r(mar_p / float(67))
    mar_n = np.where(mar_f > 0, 0, mar_f)
    # mar_n [mar_n > 0.0] = 0.0
    mar_colors_n = plt.cm.winter_r(mar_n / float(-65))

    print(mar_p)
    print(mar_n)
    print(f_c)
    # '''
    # mar_colors = []
    # mar_colors = [colormap(j) for j in np.linspace((mar_f.min()/gwsw_min), mar_f.max()/gwsw_max, len(mar_f))] # Okay, 10 is weird.
    # mar_colors = [colormap(j) for j in np.linspace(mar_f.max()/gwsw_max, (mar_f.min()/gwsw_min),len(mar_f))] # Okay, 10 is weird.
    # mar_ranks = ss.rankdata(
    #     mar_f,
    #     # method = 'dense'
    #     )
    # mar_recols = [mar_colors[(int(rank)-1)] for rank in mar_ranks]

    # jun
    # jun_f = f_c['jun'].astype('float')
    # jun_colors = [colormap(j) for j in np.linspace((jun_f.min()/gwsw_min), jun_f.max()/gwsw_max, len(jun_f))] # Okay, 10 is weird.
    # # jun_colors = [colormap(j) for j in np.linspace(jun_f.max()/gwsw_max, (jun_f.min()/gwsw_min),len(jun_f))] # Okay, 10 is weird.
    # jun_ranks = ss.rankdata(
    #     jun_f,
    #     # method = 'dense'
    #     )
    # jun_recols = [jun_colors[(int(rank)-1)] for rank in jun_ranks]
 
    jun_f = f_c['jun'].values
    
    jun_p = np.where(jun_f < 0., 0., jun_f)
    jun_colors_p = plt.cm.autumn_r(jun_p / float(67))
    jun_n = np.where(jun_f > 0, 0, jun_f)
    jun_colors_n = plt.cm.winter_r(jun_n / float(-65))

    #Sep
    sep_f = f_c['sep'].values    
    sep_p = np.where(sep_f < 0., 0., sep_f)
    sep_colors_p = plt.cm.autumn_r(sep_p / float(67))
    sep_n = np.where(sep_f > 0, 0, sep_f)
    sep_colors_n = plt.cm.winter_r(sep_n / float(-65))

    #Dec
    dec_f = f_c['dec'].values    
    dec_p = np.where(dec_f < 0., 0., dec_f)
    dec_colors_p = plt.cm.autumn_r(dec_p / float(67))
    dec_n = np.where(dec_f > 0, 0, dec_f)
    dec_colors_n = plt.cm.winter_r(dec_n / float(-65))


    # annual
    an_f = f_c['annual'].values    
    # an_p = np.where(an_f < 0., 0., an_f)
    an_colors_p = plt.cm.gist_rainbow(an_f/min(an_f))
    # an_n = np.where(an_f > 0, 0, an_f)
    # an_colors_n = plt.cm.winter_r(an_n / float(-65))




    fig, axes = plt.subplots(2, 2, figsize= (14, 7), sharex=True, sharey=True)
    fig.subplots_adjust(left = 0.03, right=0.98, top=0.95, bottom=0.1, hspace=0.5, wspace=1.2)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    ax2 = fig.add_axes([0.25, 0.3, 0.5, 0.4])

    ax3 = fig.add_subplot(111, frameon=False)
    ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')



    # ax2.bar(
    #     f_c.x_min, (an_f*-1*verExg),
    #     bottom = f_c.y_min,
    #     width = width * widthExg, align='center',
    #     alpha = 0.7, color = colormap_a, zorder = 3)
    # # ax2.bar(
    # #     f_c.x_min, (mar_n*-1*verExg),
    # #     bottom = f_c.y_min,
    # #     width = width * widthExg, align='center',
    # #     alpha = 0.7, color = mar_colors_n, zorder = 3)
    # ax2.imshow(np.random.random((10,10)), extent=[x_min, x_max, y_min, y_max], alpha = 0)


    # colors1 = plt.cm.binary(np.linspace(0., 1, 128))
    # colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 128))

    # combine them and build a new colormap
    combcolors = np.vstack((colormap_n, colormap_p))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', combcolors)


    # create dummy invisible image
    # (use the colormap you want to have on the colorbar)
    ax1 = plt.imshow(np.array([[-70, 0, 70]]), cmap=mymap)
    ax1.set_visible(False)
    cbaxes1 = fig.add_axes([0.35, 0.05, 0.3, 0.02])
    cb1 = plt.colorbar(
                    cax = cbaxes1, orientation="horizontal",
                    # ticks = [0.1,  4.82]
                    )
    cb1.outline.set_visible(False)
    # cb.set_ticks([])
    cb1.ax.set_yticklabels(['0.1 $m^3/day$', '4.8 $m^3/day$'])

    # cb.ax.invert_yaxis() 
    cb1.ax.set_title('Difference from\nAnnual Average Groundwater Discharge [$m^3/day$]', fontsize = 10,
                    # position=(1.05, 0.17),
                    horizontalalignment = 'center',
    #               y = 1.05,
                    # fontweight='semibold',
                    # transform=axes[0,0].transAxes
                    ) 
    # cb.set_label('$[m^3/day]$',
    #            #rotation=0,
    #            #y = 1.2, 
    #            labelpad= 0, 
    #            fontsize = 8)
    cb1.ax.tick_params(labelsize= 7)


    # ax3 = plt.imshow(np.array([[-65, 0]]), cmap=colormap_a2)
    # ax3.set_visible(False)
    # cbaxes3 = fig.add_axes([0.3, 0.5, 0.5, 0.02])
    # cb3 = plt.colorbar(
    #                 cax = cbaxes3, 
    #                 orientation="horizontal",
    #                 # ticks = [0.1,  4.82]
    #                 )
    # cb3.outline.set_visible(False)
    # # cb.set_ticks([])
    # cb3.ax.set_yticklabels(['0.1 $m^3/day$', '4.8 $m^3/day$'])

    # # # cb.ax.invert_yaxis() 
    # cb3.ax.set_title('Groundwater Discharge [$m^3/day$]', fontsize = 8,
    #                 # position=(1.05, 0.17),
    #                 horizontalalignment = 'center',
    # #               y = 1.05,
    #                 fontweight='semibold',
    #                 # transform=axes[0,0].transAxes
    #                 ) 
    # cb3.set_label('$[m^3/day]$',
    #            #rotation=0,
    #            #y = 1.2, 
    #            labelpad= 0, 
    #            fontsize = 8)
    # cb3.ax.tick_params(labelsize= 7)



    # plt.suptitle('Average Daily Groundwater Discharge', fontsize = 10, fontweight='bold')

    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['left'].set_visible(False)
    axes[0, 0].spines['bottom'].set_visible(False)

    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['left'].set_visible(False)
    axes[0, 1].spines['bottom'].set_visible(False)


    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['left'].set_visible(False)
    axes[1, 0].spines['bottom'].set_visible(False)

    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['left'].set_visible(False)
    axes[1, 1].spines['bottom'].set_visible(False)

    # axes[0, 0].xaxis.tick_top()
    axes[0, 0].xaxis.set_ticks_position('top')
    axes[0, 1].xaxis.set_ticks_position('top')
    axes[1, 0].get_xaxis().set_visible(False)
    axes[1, 1].get_xaxis().set_visible(False)
    axes[0, 1].get_yaxis().set_visible(False)
    axes[1, 1].get_yaxis().set_visible(False)
 
    axes[0, 0].tick_params(axis='both', labelsize=7)
    axes[0, 0].tick_params(axis='y', rotation=90)
    axes[1, 0].tick_params(axis='both', labelsize=7)
    axes[1, 0].tick_params(axis='y', rotation=90)
    axes[0, 1].tick_params(axis='both', labelsize=7)
    axes[1, 1].tick_params(axis='both', labelsize=7)

    ####
    for rv in (riv1.shapeRecords()):
        rx = [i[0] for i in rv.shape.points[:]]
        ry = [i[1] for i in rv.shape.points[:]]
        axes[0, 0].plot(rx, ry, lw = 0.5, c = 'c')
        axes[0, 1].plot(rx, ry, lw = 0.5, c = 'c')
        axes[1, 0].plot(rx, ry, lw = 0.5, c = 'c') 
        axes[1, 1].plot(rx, ry, lw = 0.5, c = 'c')
        ax2.plot(rx, ry, lw = 0.5, c = 'c')

    # Draw subbasin
    for sub in geol.shapeRecords():
        sx = [i[0] for i in sub.shape.points[:]]
        sy = [i[1] for i in sub.shape.points[:]]
        axes[0, 0].plot(sx, sy, lw = 0.5, c = 'm')
        axes[0, 1].plot(sx, sy, lw = 0.5, c = 'm')
        axes[1, 0].plot(sx, sy, lw = 0.5, c = 'm') 
        axes[1, 1].plot(sx, sy, lw = 0.5, c = 'm')
        ax2.plot(sx, sy, lw = 0.5, c = 'm')

    # axes[0, 0].axis('off')
    # axes[0, 1].axis('off')
    # axes[1, 0].axis('off')


    axes[0, 0].bar(
        f_c.x_min, (mar_p*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = mar_colors_p, zorder = 3)
    axes[0, 0].bar(
        f_c.x_min, (mar_n*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = mar_colors_n, zorder = 3)
    axes[0, 0].imshow(np.random.random((10,10)), extent=[x_min, x_max, y_min, y_max], alpha = 0)

    axes[0, 1].bar(
        f_c.x_min, (jun_p*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = jun_colors_p, zorder = 3)
    axes[0, 1].bar(
        f_c.x_min, (jun_n*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = jun_colors_n, zorder = 3)
    axes[0, 1].imshow(np.random.random((10,10)), extent=[x_min, x_max, y_min, y_max], alpha = 0)

    axes[1, 0].bar(
        f_c.x_min, (sep_p*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = sep_colors_p, zorder = 3)
    axes[1, 0].bar(
        f_c.x_min, (sep_n*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = sep_colors_n, zorder = 3)
    axes[1, 0].imshow(np.random.random((10,10)), extent=[x_min, x_max, y_min, y_max], alpha = 0)

    axes[1, 1].bar(
        f_c.x_min, (dec_p*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = dec_colors_p, zorder = 3)
    axes[1, 1].bar(
        f_c.x_min, (dec_n*-1*verExg),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = dec_colors_n, zorder = 3)
    axes[1, 1].imshow(np.random.random((10,10)), extent=[x_min, x_max, y_min, y_max], alpha = 0)


    ax2.bar(
        f_c.x_min, (an_f*-1*1.5),
        bottom = f_c.y_min,
        width = width * widthExg, align='center',
        alpha = 0.7, color = an_colors_p, zorder = 10)
    # ax2.bar(
    #     f_c.x_min, (mar_n*-1*verExg),
    #     bottom = f_c.y_min,
    #     width = width * widthExg, align='center',
    #     alpha = 0.7, color = mar_colors_n, zorder = 3)

    ax2.patch.set_facecolor('y')
    ax2.patch.set_alpha(0.1)
    ax2.imshow(np.random.random((10, 10)), extent=[x_min-1500, x_max+1000, y_min, y_max+4000],
        alpha=0)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    # # Create a Rectangle patch
    # rect = mpatches.Rectangle(
    #     (50,100),40,30,linewidth=1,edgecolor='r',facecolor='none',
    #     transform=ax2.transAxes)

    # # Add the patch to the Axes
    # ax2.add_patch(rect)
    plt.savefig('sg_int2.png', dpi=600)    
    plt.show()
    # [x_min, x_max, y_min, y_max]
    # '''



# drought_pcp_file()
# streamflow_m()
# wt_d()
data_analysis()
