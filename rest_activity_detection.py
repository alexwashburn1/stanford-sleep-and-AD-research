''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo    # debug import
import os

def read_input_data(filename):
    """
    Reads in a SINGLE cwa file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath+filename)

    return raw

# define a layout, for graphing
layout = go.Layout(
    title="Rest/Activity detection",
    xaxis=dict(title="Date time"),
    yaxis=dict(title="Counts/period"),
    showlegend=False)


# Cole-Kripke sleep detection analysis method (https://pubmed.ncbi.nlm.nih.gov/1455130/)
def Cole_Kripke_sleep_wake(raw_input_file):
    """
    will generate a figure with annotated periods of rest and periods of activity
    :param raw_input_file: the input file containing actigraphy data
    :return: no return for this
    """
    CK = raw_input_file.CK(threshold=0.1)

    # update the layout
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);


    # set up the figure
    sleep_wake_fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data', mode='markers'),
        go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK', mode='markers')
    ], layout=layout)

    # generate fig
    pyo.plot(sleep_wake_fig)


def roenneberg_sleep_wake(raw):
    roenneberg = raw.Roenneberg()
    roenneberg_thr = raw.Roenneberg(threshold=0.25, min_seed_period='15min')

    # update the layout
    layout.update(yaxis2=dict(overlaying='y', side='right'), showlegend=True);

    roenneberg_fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=roenneberg_thr.index.astype(str), y=roenneberg_thr, yaxis='y2', name='Roenneberg')
    ], layout=layout)

    roenneberg_fig.show()

def sadeh_sleep_wake(raw):
    sadeh = raw.Sadeh()

    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);

    fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=sadeh.index.astype(str), y=sadeh, yaxis='y2', name='Sadeh')
    ], layout=layout)

    fig.show()

def scripps_sleep_wake(raw):
    scripps = raw.Scripps(threshold=0.05)

    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);

    fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=scripps.index.astype(str), y=scripps, yaxis='y2', name='Scripps')
    ], layout=layout)

    fig.show()

def crespo_sleep_wake(raw):
    crespo = raw.Crespo()

    # only show sleep > 6 hrs?
    crespo_6h = raw.Crespo(alpha='6h')
    crespo_zeta = raw.Crespo(estimate_zeta=True)

    # update the figure layout
    layout.update(yaxis2=dict(overlaying='y', side='right'), showlegend=True);

    crespo_fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=crespo_zeta.index.astype(str), y=crespo_zeta, yaxis='y2', name='Crespo (Automatic)')
    ], layout=layout)

    crespo_fig.show()

    aot = raw.Crespo_AoT()
    print('aot: ', aot)



'''FUNCTION CALLS'''
raw = read_input_data('67180_0003900497-timeSeries.csv.gz')
#Cole_Kripke_sleep_wake(raw)
roenneberg_sleep_wake(raw)
#crespo_sleep_wake(raw)
#sadeh_sleep_wake(raw)
#scripps_sleep_wake(raw)


