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
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath+filename)

    # Retrieve header information from the data set
    file_name = raw.name
    print('file name: ' + file_name)

    file_start_time = raw.start_time
    print('file start time: ')
    print(file_start_time)

    file_data_acquisition_duration = raw.duration()
    print('data acquisition duration: ')
    print(file_data_acquisition_duration)

    device_serial_number = raw.uuid
    print('device serial number: ')
    print(device_serial_number)

    acquisition_frequency = raw.frequency
    print('acquisition frequency: ')
    print(acquisition_frequency)

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
    CK = raw_input_file.CK()

    # update the layout
    layout.update(yaxis2=dict(title='Classification', overlaying='y', side='right'), showlegend=True);

    # set up the figure
    sleep_wake_fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=CK.index.astype(str), y=CK, yaxis='y2', name='CK')
    ], layout=layout)

    # generate fig
    pyo.plot(sleep_wake_fig)


def sadeh_scripps_sleep_wake(raw_input_file):  # TODO - not working at the moment

    sadeh = raw_input_file.Sadeh()
    scripps = raw_input_file.Scripps()

    # set up the figure
    sadeh_scripps_fig = go.Figure(data=[
        go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Data'),
        go.Scatter(x=sadeh.index.astype(str), y=sadeh, yaxis='y2', name='Sadeh'),
        go.Scatter(x=scripps.index.astype(str), y=scripps, yaxis='y2', name='Scripps')
    ], layout=layout)

    # generate fig
    pyo.plot(sadeh_scripps_fig)




'''FUNCTION CALLS'''
raw = read_input_data('79012_0003900559-timeSeries.csv.gz')  # the file with disrupted sleep patterns potentially present
Cole_Kripke_sleep_wake(raw)
#sadeh_scripps_sleep_wake(raw)


