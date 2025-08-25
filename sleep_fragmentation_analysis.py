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
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw

# create a layout for plotting
layout = go.Layout(title="",xaxis=dict(title=""), showlegend=False)

# Quantification of rest-to-activity transition probability
def rest_to_activity_transition(raw):
    """
    look at the distribution of the transition probabilities as a function of the length of the sustained rest periods.
    Plot the probability of a rest-> activity transition against previous time of sustained rest. Higher probability
    indicates a higher sleep fragmentation.
    :param raw: the raw data file with actigraphy data
    :return:
    """
    ### pRA graph ###
    pRA, pRA_weights = raw.pRA(0.01, start='00:00:00', period='8H')

    # update layout
    layout.update(title="Rest->Activity transition probability",xaxis=dict(title="Time [min]"), showlegend=False);

    fig = go.Figure(data=go.Scatter(x=pRA.index, y=pRA, name='', mode='markers'), layout=layout)

    fig.show()


    ### pAR graph ###
    pAR, pAR_weights = raw.pAR(0.01, start='00:00:00', period='8H')

    # update layout
    layout.update(title="Activity -> Rest transition probability", xaxis=dict(title="Time [min]"), showlegend=False);

    fig = go.Figure(data=go.Scatter(x=pAR.index, y=pRA, name='', mode='markers'), layout=layout)

    fig.show()

'''FUNCTION CALLS'''
# 79012_0003900559
raw = read_input_data('79036_0000000504-timeSeries.csv.gz')
rest_to_activity_transition(raw)

