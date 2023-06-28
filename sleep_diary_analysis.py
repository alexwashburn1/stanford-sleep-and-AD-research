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
    Reads in a SINGLE csv.gz file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw

def plot_sleep_log(raw, sleep_diary):
    """
    plots the sleep log, with the crespo sleep detection algorithm and cole-kripke for comparison.
    :param raw:
    :param sleep_diary:
    :return:
    """
    # define a layout
    layout = go.Layout(
        title="Actigraphy data",
        xaxis=dict(title="Date time"),
        yaxis=dict(title="Counts/period"),
        shapes=raw.sleep_diary.shapes(),
        showlegend=False
    )

    # add a crespo layer
    crespo = raw.Crespo()

    crespo_6h = raw.Crespo(alpha='6h')
    crespo_zeta = raw.Crespo(estimate_zeta=True)
    CK = raw.CK(threshold=0.1)

    raw.sleep_diary.shaded_area

    raw.sleep_diary.shaded_area['opacity'] = 1

    layout.update(yaxis2=dict(overlaying='y', side='right'), showlegend=True, shapes=raw.sleep_diary.shapes());

    fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=crespo_zeta.index.astype(str), y=crespo_zeta, yaxis='y2', name='Crespo (Automatic)'),
        go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK', mode='markers')
    ], layout=layout)

    fig.show()





'''FUNCTION CALLS'''
id = '78203_0000000534'
sleep_diary_filename = id+'.ods'
timeseries_filename = id+'-timeSeries.csv.gz'

# read in the raw data file
raw = read_input_data(timeseries_filename)

# try to read in the sleep diary
fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/'
sleep_diary = raw.read_sleep_diary(fpath + sleep_diary_filename)
print('name: ', raw.sleep_diary.name)
print('sleep diary: ', raw.sleep_diary.diary)

print(plot_sleep_log(raw, sleep_diary))

