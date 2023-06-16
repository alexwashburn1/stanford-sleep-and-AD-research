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
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw


def cosinor_analysis_on_file(raw):
    """

    :param raw: the raw input file to conduct cosinor analysis on.
    :return:
    """

    # create a cosinor object
    cosinor = Cosinor()
    cosinor.fit_initial_params['Period'].value = 1440  # set period value
    cosinor.fit_initial_params['Period'].vary = False
    cosinor.fit_initial_params.pretty_print()  # default values for cosinor analysis

    # perform cosinor analysis
    results = cosinor.fit(raw.data, verbose=True)  # Set verbose to True to print the fit output

    # access the best fit parameter values
    results.params['Mesor'].value

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
if np.isnan(raw.data).any():
    print('entered')
    # Handle NaN values in raw.data (e.g., by removing or filling them)
    raw.data = raw.data.dropna()  # Remove rows with NaN values


# cosinor_analysis_on_file(raw)
