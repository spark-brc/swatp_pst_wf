# import plotly.graph_objects as go
import os
import pandas as pd
# import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# NOTE: sankey

from swatp_pst.handler import SWATp


def fig():

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = ["A1", "A2", "B1", "B2", "C1", "C2"]
        
        ),
        link = dict(
        source = [0, 0, 1, 2, 3, 3], 
        target = [2, 3, 3, 4, 4, 5],
        value =  [8, 2, 4, 8, 4, 2],

    ))])

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10,width=600, height=400)
    fig.show()



def read_data(wd):
    os.chdir(wd)
    df = pd.read_csv('hf_data.csv')
    '''
    unique_sts = pd.concat([df.source, df.target]).unique().tolist()
    sts_dic = {k: v for v, k in enumerate(unique_sts)}
    # replace name to id
    for i in range(len(df)):
        nam = df.iloc[i,0]
        val = sts_dic.get(nam)
        df.iloc[i,0] = val
    for i in range(len(df)):
        nam = df.iloc[i,1]
        val = sts_dic.get(nam)
        df.iloc[i,1] = val
    '''
    return df
    


def fig_test(wd):
    all_links = read_data(wd)
    #for using with 'label' parameter in plotly 
    #https://sparkbyexamples.com/pandas/pandas-find-unique-values-from-columns
    unique_source_target = list(pd.unique(all_links[['source', 'target']].values.ravel('K')))

    #for assigning unique number to each source and target
    mapping_dict = {k: v for v, k in enumerate(unique_source_target)}

    #mapping of full data
    all_links['source'] = all_links['source'].map(mapping_dict)
    all_links['target'] = all_links['target'].map(mapping_dict)

    #converting full dataframe as list for using with in plotly
    links_dict = all_links.to_dict(orient='list')

    nodes = np.unique(all_links[["source", "target"]], axis=None)
    nodes = pd.Series(index=nodes, data=range(len(nodes)))
    # colors = [
    #         px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    #         for i in nodes.loc[all_links["source"]]
    #         ]
    
    colors = [
            px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i in nodes.loc[all_links["source"]]
            ]
    
    testcolors = [px.colors.hex_to_rgb(co) for co in colors]
    testcolors2 = [f"rgba{list(a).insert(2, '0.5')}" for a in testcolors]

    print(testcolors2)

    #Sankey Diagram Code 
    fig = go.Figure(data=[go.Sankey(
        # arrangement = "snap",
        node = dict(
                    pad = 30,
                    thickness = 10,
                    line = dict(color = "black", width = 0.5),
                    label = [""]*13,
                    # color = links_dict["node_color"]
                    ),
        link = dict(
                source = links_dict["source"],
                target = links_dict["target"],
                value = links_dict["value"],
                # color=['red', 'blue', 'red', 'yellow', 'green', 'blue'],
                # color = links_dict["link_color"],
                # opacity = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                color = colors
        ))])




    fig.update_layout(font_size=10,width=1000, height=500)
    fig.show()



def fig_ttt_fp(wd):
    fig = plt.figure(
        figsize=(10, 8)
        )
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],)
    sankey = Sankey(ax=ax, 
                    scale=0.01, 
                    # offset=10, 
                    # head_angle=180,
                    # format='%.0f',
                    unit=None,
                    # gap=1
                    )
    sankey.add(flows=[1175, 340, 117, 50, -570,  -262, -898],
        # labels=['pcp', 'irr', 'surq_r
        # unon', 'latq_runon'
        # , 'perc', 'et', 'yld'],
        orientations=[1, 1, 1, 1, 1, -1, 0],
        trunklength=10,
        pathlengths=[10, 6, 1, 1, 5, 1, 3],
        fc="C1"
        # rotation=90
        ).finish()
    # plt.title("The default settings produce a diagram like this.")
    plt.show()

def fig_ttt_up(wd):
    fig = plt.figure(
        figsize=(10, 8)
        )
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                        # title="Flow Diagram of a Widget"
                        )
    sankey = Sankey(ax=ax, 
                    scale=0.01, 
                    # offset=10, 
                    # head_angle=180,
                    # format='%.0f',
                    unit=None,
                    # gap=1
                    )
    sankey.add(flows=[1174, 85, -432,  -154, -10.6, -2.7, -448, ],
        # labels=['pcp', 'irr', 'surq_r
        # unon', 'latq_runon'
        # , 'perc', 'et', 'yld'],
        orientations=[1, 1, 1, 0, -1,-1,-1],
        trunklength=10,
        pathlengths=[10, 1, 5, 2, 1,2,1],
        fc="C2"
        # rotation=90
        ).finish()
    # plt.title("The default settings produce a diagram like this.")
    plt.show()

def fig_ttt_aq01(wd):
    fig = plt.figure(
        figsize=(10, 8)
        )
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                        # title="Flow Diagram of a Widget"
                        )
    sankey = Sankey(ax=ax, 
                    scale=0.01, 
                    # offset=10, 
                    # head_angle=180,
                    # format='%.0f',
                    unit=None,
                    # gap=1
                    )
    sankey.add(flows=[448, -386],
        # labels=['pcp', 'irr', 'surq_r
        # unon', 'latq_runon'
        # , 'perc', 'et', 'yld'],
        orientations=[1, 0],
        trunklength=5,
        pathlengths=[10, 1],
        fc="C0"
        # rotation=90
        ).finish()
    # plt.title("The default settings produce a diagram like this.")
    plt.show()


def fig_ttt_aq02(wd):
    fig = plt.figure(
        figsize=(10, 8)
        )
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                        # title="Flow Diagram of a Widget"
                        )
    sankey = Sankey(ax=ax, 
                    scale=0.01, 
                    # offset=10, 
                    # head_angle=180,
                    # format='%.0f',
                    unit=None,
                    # gap=1
                    )
    sankey.add(flows=[9, 386, -382],
        # labels=['pcp', 'irr', 'surq_r
        # unon', 'latq_runon'
        # , 'perc', 'et', 'yld'],
        orientations=[1, 0, 0],
        trunklength=5,
        pathlengths=[1, 25, 1],
        fc="C0"
        # rotation=90
        ).finish()
    # plt.title("The default settings produce a diagram like this.")
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
    #                     title="Flow Diagram of a Widget")
    # sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
    #                 format='%.0f', unit='%')
    # sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
    #         labels=['', '', '', 'First', 'Second', 'Third', 'Fourth',
    #                 'Fifth', 'Hurray!'],
    #         orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
    #         pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
    #                         0.25],
    #         patchlabel="Widget\nA")  # Arguments to matplotlib.patches.PathPatch
    # diagrams = sankey.finish()
    # diagrams[0].texts[-1].set_color('r')
    # diagrams[0].text.set_fontweight('bold')
    # plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
    # flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
    # sankey = Sankey(ax=ax, unit=None)
    # sankey.add(flows=flows, label='one',
    #         orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
    # sankey.add(flows=[-0.25, 0.15, 0.1], label='two',
    #         orientations=[-1, -1, -1], prior=0, connect=(0, 0))
    # diagrams = sankey.finish()
    # diagrams[-1].patch.set_hatch('/')
    # plt.legend()
    # plt.show()


    # Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
    #     labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
    #     orientations=[-1, 1, 0, 1, 1, 1, 0, -1],
    # #        color=['k','r']
    #     ).finish()
    # plt.title("The default settings produce a diagram like this.")
    # plt.show()


# def plot_tot():
if __name__ == '__main__':
    wd = "D:\\spark_papers\\2024_Jeong_et_al_swatp-paddy"
    fig_ttt_up(wd)
    wd = "D:\\Projects\\Watersheds\\osu\\opt_3rd\\calibrated_model_v02"
    m1 = SWATp(wd)
    m1.get_lu_hf_wb()

    

