''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo    # debug import
import os
from pyActigraphy.analysis import Cosinor  # for cosinor analysis

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

def mask_out_inactivity(raw):
    """
    masks out periods of inactivity for a given raw file.
    :param raw: the raw cwa file
    :return:
    """

    # create a mask automatically
    raw.frequency

    # The duration corresponds either to the minimal number of inactive epochs (ex: duration=120)
    # or the minimal length of the inactive period (duration='2h00min')
    raw.create_inactivity_mask(duration='0h30min') #TODO - can change the duration (2 hr typical)

    # find inactivity length
    print(raw.inactivity_length)

    # visualize the mask
    layout = go.Layout(title="Data mask", xaxis=dict(title="Date time"), yaxis=dict(title="Mask"), showlegend=False)
    fig = go.Figure(data=go.Scatter(x=raw.mask.index.astype(str), y=raw.mask), layout=layout)
    fig.show()

    # apply the mask
    print('before mask: ')
    print(raw.IS())
    raw.mask_inactivity = True
    print('after mask: ')
    print(raw.IS())

    # for the hell of it, try cosinor analysis on the new raw file
    # create a cosinor object
    cosinor = Cosinor()
    cosinor.fit_initial_params['Period'].value = 1440  # set period value
    cosinor.fit_initial_params['Period'].vary = False
    cosinor.fit_initial_params.pretty_print()  # default values for cosinor analysis

    # perform cosinor analysis
    print('reached here: ')
    results = cosinor.fit(raw.data, verbose=True)  # Set verbose to True to print the fit output
    print('reached second point')


'''FUNCTION CALLS'''
raw = read_input_data('79012_0003900559-timeSeries.csv.gz')

mask_out_inactivity(raw)


