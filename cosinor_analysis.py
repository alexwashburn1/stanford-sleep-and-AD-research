''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import os
from pyActigraphy.analysis import Cosinor  # for cosinor analysis


def read_input_data(filename):
    """
    Reads in a SINGLE cwa file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename, impute_missing = True)

    return raw

# create a layout for graphing
layout = go.Layout(title="Cosinor Analysis",xaxis=dict(title="Counts/period"), showlegend=True)

def cosinor_analysis_on_file(raw):
    """
    Fit a cosine trend line to the file to th
    :param raw: the raw input file to conduct cosinor analysis on.
    :return:
    """

    # create a cosinor object
    cosinor = Cosinor()
    cosinor.fit_initial_params['Period'].value = 2880  # set period value
    cosinor.fit_initial_params['Period'].vary = False
    cosinor.fit_initial_params.pretty_print()  # default values for cosinor analysis

    # perform cosinor analysis
    print('reached here: ')
    results = cosinor.fit(raw.data, verbose=True)  # Set verbose to True to print the fit output
    print('reached second point')

    # access the best fit parameter values
    results.params['Mesor'].value

    # update the layout
    layout.update(title="Cosinor Analysis", xaxis=dict(title="Time [min]"), yaxis = dict(title="Counts/period"), showlegend=True);

    ## FIT RESULT VISUALIZATION
    best_fit = cosinor.best_fit(raw.data, results.params)
    fig = go.Figure(
        data=[
            go.Scatter(x=raw.data.index.astype(str), y=raw.data, name='Raw data'),
            go.Scatter(x=best_fit.index.astype(str), y=best_fit, name='Best fit')
        ]
    )

    fig.show()  # actually output the figure


'''FUNCTION CALLS'''
raw = read_input_data('79012_0003900559-timeSeries.csv.gz')
print(type(raw))
# filter out na values TODO work in progress
# if np.isnan(raw.data).any():
#     print('entered')
#     # Handle NaN values in raw.data (e.g., by removing or filling them)
#     raw.data = raw.data.dropna() # TODO figure out how to filter out NA values from the data set.

# check for NaN values in the data set
print("Raw data has NaNs: ", raw.raw_data.hasnans)

index_with_number = 0
print("entry with number", raw.raw_data[index_with_number])

index_without_number = 7132
print("entry without number", raw.raw_data[index_without_number])

cosinor_analysis_on_file(raw)
