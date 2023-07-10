'''IMPORTS'''
import sys
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyActigraphy.analysis import LIDS   #LIDS tools import
import plotly.graph_objects as go

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
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw

def LIDS_functionality_test(LIDS_obj, raw):
    """
    Test the functionality of LIDS on a single raw file, for now.
    :param LIDS_obj: the LIDS object to test the functionality for.
    :return:
    """

    # transform LIDS data
    lids_transformed = LIDS_obj.lids_transform(ts=raw.data)
    LIDS_obj.lids_fit(lids=lids_transformed, nan_policy='omit', verbose=False)

    # statistics
    lids_period = LIDS_obj.lids_fit_results.params['period']
    print('LIDS period: ', lids_period)

    correlation_factor = LIDS_obj.lids_pearson_r(lids_transformed)
    print('pearson correlation factor: ', correlation_factor)

    lids_mri = LIDS_obj.lids_mri(lids_transformed)
    print('Munich Rhythmicity Index: ', lids_mri)

    lids_phases = LIDS_obj.lids_phases(lids_transformed)
    print('Phases ', lids_phases)

    # append the lids data to a list, since we want a list of series
    lids_series_list = []
    lids_series_list.append(lids_transformed)
    #lids_summary = LIDS_obj.lids_summary(lids_series_list, verbose=False)
    #print('lids summary: ', lids_summary)


    # plot the LIDS transformed data
    plt.plot(lids_transformed)

    plt.show()

def extract_bout(raw, start, end):
    """
    Filters to extract the timeseries data for a bout given the start and end times for the bout.
    :param raw:
    :param start:
    :param end:
    :return:
    """
    # filter Pandas Series by timestamp
    bout = raw.loc[start:end]
    return bout

def extract_activity_data(bout):
    """
    Extracts the activity data from a sleep period (bout)
    :param bout: Pandas Series of (timestamp, activity value) tuples
    :return: Pandas Series of  activity values
    """
    # given Panda series of (timestamp, activity value) tuples, extract the activity values
    df = pd.DataFrame(bout.tolist())
    return df[0]   # return the activity values


def find_bouts(lids_obj, raw):
    # extract sleep periods from the raw data from one file, via Crespo's AoT method
    # filter the extracted sleep periods for sleep periods too short or too long
    # transform activity data for each sleep period to inactivity data

    (wakes, sleeps) = raw.Crespo_AoT()
    print('crespo AOT: ', raw.Crespo_AoT())
    # iterate through the wakes and sleeps and find the on and off times
    sleep_wakes = []
    for i in range(len(wakes)):
        sleep_wakes.append((sleeps[i], wakes[i]))  # append the tuple of the on and off times
    # iterate through the sleep_wakes and find the duration of each sleep
    sleep_bouts = []
    for i in range(len(sleep_wakes)):
        sleep_bouts.append(extract_bout(raw.data, sleep_wakes[i][0], sleep_wakes[i][1]))

    sleep_bouts_filtered = lids_obj.filter(ts=sleep_bouts, duration_min='6H')
    #plot_bouts(sleep_bouts_filtered)
    bouts_transformed = []

    for i in range(len(sleep_bouts_filtered)):
        transform = True
        if transform:
            bouts_transformed.append(lids_obj.lids_transform(ts=sleep_bouts_filtered[i]))
        else:   # don't transform
            bouts_transformed.append(sleep_bouts_filtered[i])

    #plot_bouts(bouts_transformed)
    return bouts_transformed

def find_shortest_series(series):
    """

    :param series: list of Series of sleep activity values
    :return: length of shortest list
    """
    min_length = sys.maxsize
    for i in range(len(series)):
        if len(series[i]) < min_length:
            min_length = len(series[i])  # update the min length
    return min_length

def mean_of_bouts(bouts):
    # extract the activity data from the bouts
    activity_data = []
    for i in range(len(bouts)):  # iterate through the bouts
        activity_data.append(extract_activity_data(bouts[i]))  # append the activity data from each bout
    # find the shortest series
    length_of_shortest_series = find_shortest_series(activity_data)
    print("length of shortest series: ", length_of_shortest_series)
    # trim the series to the shortest series
    for i in range(len(activity_data)):
        activity_data[i] = activity_data[i][:length_of_shortest_series]
    # convert the activity data to a numpy array
    activity_data = np.array(activity_data)
    # take the mean of the activity data
    activity_data_mean = np.mean(activity_data, axis=0)
    return activity_data_mean

def per_file(filename):
    raw = read_input_data(filename)
    bouts = find_bouts(test_lids_obj, raw)
    activity_data_mean = mean_of_bouts(bouts)

    return (activity_data_mean, len(bouts))


def set_up_plot(filenames):
    """
    Takes the mean over several bouts for several files. Filters all files according to the shortest record. Plots.
    :param filenames: the filenames to include in the ultimate LIDS graph
    :return:
    """
    file_means = []
    total_bouts = 0
    for i in range(len(filenames)):
        (file_mean, n_bouts_in_file) = per_file(filenames[i])
        file_means.append(file_mean)  # append the mean of the activity data for each file
        total_bouts += n_bouts_in_file

    length_of_shortest_array = find_shortest_series(file_means)
    print("length of shortest array: ", length_of_shortest_array)

    for i in range(len(file_means)):
        file_means[i] = file_means[i][:length_of_shortest_array]

    plt.figure()
    # set label for x axis
    plt.xlabel('time (30s)')
    # set label for y axis
    plt.ylabel('inactivity')
    # set title
    n_files = len(file_means)

    file_mean = np.mean(file_means, axis=0)
    plt.title(f'mean of {total_bouts} bouts from {n_files} files')
    plt.plot(file_mean)
    plt.show()


def debug_LIDS_graph(raw):
    """
    Debugging to confirm that the LIDS data is not inverted.
    :param raw:
    :return:
    """

    #crespo = raw.Crespo()

    layout  = go.Layout(title="Rest/Activity detection",xaxis=dict(title="Date time"), yaxis=dict(title="Counts/period"), showlegend=False)

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


''' FUNCTION CALLS '''

filename = '67067_0000000131-timeSeries.csv.gz'
raw = read_input_data(filename)

# DEBUG
#debug_LIDS_graph(raw)

# Create a LIDS object
test_lids_obj = LIDS()

# test lids functionality
#LIDS_functionality_test(test_lids_obj, raw)


#### LIDS GRAPHICAL ANALYSIS ####
directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

filenames = [filename for filename in os.listdir(directory) if filename.endswith('timeSeries.csv.gz')]
set_up_plot(filenames[1:2])











