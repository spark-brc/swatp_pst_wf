from swatp_pst import analyzer
from swatp_pst import handler
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


def violinplot():
    wd = "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    df = m1.get_hru_area()
    dft = m1.get_hru_lsu_df()

    wd2 = "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m2 = handler.SWATp(wd2)
    df2 = m2.get_hru_area()
    dft2 = m2.get_hru_lsu_df()
    
    sitenam = "Dawhenya"
    sitenam2 = "Mun"
    # print(type(df))
    stats = analyzer.SWATp.create_stat_df(df)
    stats2 = analyzer.SWATp.create_stat_df(df2)
    print(stats)
    print(stats2)

    # df = [x for x in df if x < 300]
    fig, axes = plt.subplots(1, 2, figsize=(3, 5), sharey=True)
    axes[0] = analyzer.SWATp.violin_hru_lsu(axes[0], df, sitenam)
    axes[1] = analyzer.SWATp.violin_hru_lsu(axes[1], df2, sitenam2, 'C1')
    axes[0].set_ylim(-50, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'hru_lsu_size.png'), bbox_inches='tight', dpi=300)  
    plt.show()


def pie_landuse():
    wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    df = m1.get_landuse()
    print(df.area.sum())
    wd2 = "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m2 = handler.SWATp(wd2)
    df2 = m2.get_landuse()
    print(df2.area.sum())
    
    sitenam = "Dawhenya"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0] = analyzer.SWATp.pie_landuse(axes[0], df)
    axes[1] = analyzer.SWATp.pie_landuse(axes[1], df2)

    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'pie.png'), bbox_inches='tight', dpi=300)
    plt.show()    

def weather_irr():
    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    wd =  "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    df = m1.monthly_weather_irr()
    # print(df.index)
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax2 = ax.twinx()
    # analyzer.SWATp.bar_weather_irr(ax, ax1, ax2, df)
    analyzer.SWATp.bar_monthly_weather(ax, ax1, df)
    ax.set_ylim(0, 400)
    ax.invert_yaxis()
    ax1.set_ylim(20, 40)
    ax2.set_ylim(0, 1500)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'weather.png'), bbox_inches='tight', dpi=300)   
    # plt.legend()
    plt.show()

def weather_irr_mun():
    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    wd =  "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    df = m1.monthly_weather_irr()
    # print(df.index)
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    # analyzer.SWATp.bar_weather_irr(ax, ax1, ax2, df)
    analyzer.SWATp.bar_monthly_weather(ax, ax1, df)
    ax.set_ylim(0, 400)
    ax.invert_yaxis()
    ax1.set_ylim(15, 40)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'weather_mun.png'), bbox_inches='tight', dpi=300)   
    # plt.legend()
    plt.show()



def get_wb():
    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    wd =  "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    
    fields = ["wateryld", "perc", "et", "sw_ave", "latq_runon"]
    for f in fields:
        m1.get_lu_mon(f)


def plot_yield():
    wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    m1 = handler.SWATp(wd)
    obdfile = "crop_yr_obd.csv"
    df = m1.get_crop_yld_sim_obd(obdfile)
    df = df.loc[2015:, :]
    fig, ax = plt.subplots()
    analyzer.SWATp.plot_yield(ax, df)
    ax.set_ylim(0, 9)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)
    plt.tight_layout()
    # plt.legend()
    plt.savefig(os.path.join(wd, 'crop.png'), bbox_inches='tight', dpi=300)  
    plt.show()


def plot_yield2():
    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    wd =  "D:\\Projects\\Watersheds\\Mun\\Mun_river_082024\\Scenarios\\Default\\TxtInOut"
    m1 = handler.Paddy(wd)
    obdfile = "crop_yr_obd.csv"
    # name = "Dawhenya"
    name = "2020"
    # df = m1.get_crop_yld_sim_obd(obdfile)
    fig, ax = plt.subplots(figsize=(3,5))
    df = m1.get_crop_yld()
    analyzer.SWATp.violin_hru_lsu(ax, df.loc[:, "yield"], name)
    ax.scatter([1], [2187], marker="x", color='m', label="Sticky rice, 2024", zorder=3, s=50)
    ax.scatter([1], [2500], marker="v", color='g', label="Jasmine rice, 2024",zorder=3, s=50)
    # ax.set_ylim(0, 9)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2,0,1]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)
    plt.tight_layout()
    # plt.legend()
    plt.savefig(os.path.join(wd, 'crop_mun.png'), bbox_inches='tight', dpi=300)  
    plt.show()


if __name__ == '__main__':
    # wd =  "D:\\Projects\\Watersheds\\Ghana\\Analysis\\dawhenya\\prj05_paddy\\Scenarios\\Default\\TxtInOut"
    # m1 = handler.Paddy(wd)

    weather_irr_mun()

