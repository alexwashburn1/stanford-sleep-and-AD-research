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

def read_files_by_batch(file_identifier):
    """
    Read in multiple cwa files in using a common identifier.
    :param file_identifier: a string containing the text that all files we want to read contain, as well as a wildcard.
    :return: the batch of raw files that were read in
    """
    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/'

    #  Read test files
    raw_batch = pyActigraphy.io.read_raw(fpath+file_identifier, reader_type='BBA')

    # Check how many files have been read
    print('number of files in the batch: ')
    print(len(raw_batch.readers))

    # print the duration of each file
    print('duration of each file in the batch: ')
    print(raw_batch.duration())

    # print the inter-daily stability corresponding to each cwa file:
    print("mean inter-daily stability of each cwa file:")
    print(raw_batch.ISm())

    # print the inter-daily stability corresponding to each period in each file:
    print('inter-daily stability corresponding to each period in each file: ')
    print(raw_batch.ISp(period = '6D'))

    # calculate and print the relative amplitude of each cwa file timeseries:
    print('relative amplitude of each cwa file: ')
    print(raw_batch.RA())



    return raw_batch

# create a layout for all future plotting
layout = go.Layout(
    title = "Actigraphy data",
    xaxis = dict(title="Date time"),
    yaxis = dict(title="Counts/period"),
    showlegend = False
)

'''PLOTTING THE DATA - MULTI-DAY'''
def multi_day_plot(raw_data_file, layout):
    """
    generates a plot of multi-day actigraphy data
    :param raw_data_file: the raw data file for the actigraphy plot
    :param layout: a pre-defined layout for the graph
    :return: no return value
    """

    # create the figure
    fig = go.Figure(data=[go.Scatter(x=raw.data.index.astype(str), y=raw.data)], layout=layout)

    # display the figure
    pyo.plot(fig)


'''DAILY ACTIVITY PROFILE'''
def daily_activity_profile(raw_data_file, layout):
    """
    generates a daily activity profile (subset of multi day actigraphy plot)
    :param raw_data_file: the raw data file for the actigraphy plot
    :param layout: a pre-defined layout for the graph
    :return:
    """
    layout.update(title="Daily activity profile",xaxis=dict(title="Date time"), showlegend=False);
    help(raw.average_daily_activity)

    daily_profile = raw.average_daily_activity(freq='15min', cyclic=False, binarize=False) # frequency = data resampling frequency

    # create the daily activity profile figure
    daily_fig = go.Figure(data=[
        go.Scatter(x=daily_profile.index.astype(str), y=daily_profile)
    ], layout=layout)

    # plot the daily figure
    pyo.plot(daily_fig)


'''ACTIVITY ONSET AND OFFSET TIMES'''
def retrieve_activity_onset_offset(raw_data_file):
    """
    This function will retrieve the time of activity onset and offset.
    :param raw_data_file: the raw data file containing actigraphy data
    :return: a list containing the activity onset time followed by the activity offset time
    """
    activity_onset = raw.AonT(freq='15min', binarize=True)
    activity_offset = raw.AoffT(freq='15min', binarize=True)

    print('activity onset: ')
    print(activity_onset)

    print('activity offset: ')
    print(activity_offset)

    return [activity_onset, activity_offset]


'''FUNCTION CALLS'''
# raw = read_input_data('79036_0000000504-timeSeries.csv.gz')  # change the filename here
raw_batch = read_files_by_batch('*timeSeries.csv.gz')  # read in anything containing characters following wildcard
print(raw_batch.IS())
# multi_day_plot(raw, layout)
# daily_activity_profile(raw, layout)
# retrieve_activity_onset_offset(raw)






