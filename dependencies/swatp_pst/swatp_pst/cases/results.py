from swatp_pst import analyzer
from swatp_pst import handler
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker




def pie_landuse(wd):
    m1 = handler.SWATp(wd)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = analyzer.SWATp.pie_landuse(ax, m1.get_landuse())
    print(m1.get_landuse())
    plt.tight_layout()
    plt.savefig(os.path.join(wd, 'pie.png'), bbox_inches='tight', dpi=300)
    plt.show()





if __name__ == "__main__":
    wd = "d:\\Projects\\Tools\\WISE\\CoSWAT-Framework\\model-setup\\CoSWATv0.1.0\\asia-korea\\Scenarios\\Default\\TxtInOut\\"
    # pie_landuse(wd)
    # m1 = handler.SWATp(wd)
    # df = m1.get_rice_hru_info(lum_name="rice140_lum")
    # median_area = df['area'].median()
    # median_index = df[df['area'] == median_area].index
    # print(df.loc[median_index])
    # print(median_area)
    # print(df.loc[df['area'].idxmax()])
    # print(df.loc[df['area'].idxmin()])
    # max_area = df[df['name'] == df['area'].idxmax()]
    # min_area = df[df['id'] == df['area'].idxmin()]
    # print(max_area)
    # a = analyzer.Paddy(wd)
    # a.plot_paddy_daily(max_area)    
    # print(df.loc[:, ['id', 'area']])


    m1 = handler.Paddy(wd)
    

    # df = m1.read_paddy_daily(i)
    # print(f"paddy {i} area: {df['area'].sum()}")

    for i in [36, 8336, 12682]:
        df = m1.read_paddy_daily(i).loc[:, ['Precip', 'Irrig', 'Seep', 'ET', 'WeirH', 'Wtrdep', 'WeirQ', 'LAI']]
        if i == 12682:
            analyzer.Paddy(wd).plot_paddy_daily_bar(df["1980-10-01":"1980-10-10"])
        else:
            analyzer.Paddy(wd).plot_paddy_daily_bar(df["1980-09-27":"1980-10-05"])
        analyzer.Paddy(wd).plot_paddy_daily(df) 
        print(i)
    # print(df)
    '''
    '''
