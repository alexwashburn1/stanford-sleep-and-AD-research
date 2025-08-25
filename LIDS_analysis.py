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

# Create a LIDS object
lids_obj = LIDS()

# set max LIDS periods value
MAX_PERIODS = 5
max_x_value = MAX_PERIODS

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
    raw = pyActigraphy.io.read_raw_bba(fpath + filename, use_metadata_json=False)

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


def helper_unequal_sleep_wakes(lids_obj, raw, wakes, sleeps):
    """
    Helper function for find_bouts that will skip a subject if the subject has an unequal record of sleeps and wakes.
    :param lids_obj: lids object
    :param raw: actigraphy data for the subject
    :return:
    """

    if len(wakes) != len(sleeps):
        print('length wakes: ', len(wakes))
        print('length sleeps: ', len(sleeps))
        print('wakes and sleeps: ', (wakes, sleeps))
        return True
    else:
        return False

def find_bouts(lids_obj, raw):
    # extract sleep periods from the raw data from one file, via Roennebergs's AoT method
    # filter the extracted sleep periods for sleep periods too short or too long
    # transform activity data for each sleep period to inactivity data

    # use the helper function first to ensure that the number of wakes and the number of sleeps are not unequal. If they
    # are, ignore the record.

    (wakes, sleeps) = raw.Roenneberg_AoT()

    if helper_unequal_sleep_wakes(lids_obj, raw, wakes, sleeps) == True:
            # check if there is an extra wake value at the beginning
        if wakes[0] < sleeps[0]:
            wakes = wakes[1:]

    # default threshold = 0.15, according to <15% threshold used in LIDs methods (Winnebeck 2018)
    # iterate through the wakes and sleeps and find the on and off times
    sleep_wakes = []
    print('length sleeps: ', len(sleeps))
    print('length of the wakes: ', len(wakes))
    for i in range(len(wakes)):
        sleep_wakes.append((sleeps[i], wakes[i]))  # append the tuple of the on and off times
    # iterate through the sleep_wakes and find the duration of each sleep. Append to sleep_bouts.
    sleep_bouts = []
    for i in range(len(sleep_wakes)):
        sleep_bouts.append(extract_bout(raw.data, sleep_wakes[i][0], sleep_wakes[i][1]))

    sleep_bouts_filtered = lids_obj.filter(ts=sleep_bouts, duration_min='3H', duration_max='12H')

    # resample/downscale the sleep bouts to have 10 minute bins
    sleep_bouts_filtered = resample_bouts(sleep_bouts_filtered)

    #plot_bouts(sleep_bouts_filtered)
    bouts_transformed = []

    # transform only the sleep bouts that match the duration requirement.
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
    """
    Identifies the mean of the bouts for a single file.
    :param bouts:
    :return:
    """
    # extract the activity data from the bouts
    activity_data = []
    for i in range(len(bouts)):  # iterate through the bouts
        activity_data.append(extract_activity_data(bouts[i]))  # append the activity data from each bout
    # find the shortest series
    length_of_shortest_series = find_shortest_series(activity_data)
    print("length of shortest series: ", length_of_shortest_series)
    # trim the series to the shortest series
    for i in range(len(activity_data)):     # WHAT DOES THIS DO?
        activity_data[i] = activity_data[i][:length_of_shortest_series]
    # convert the activity data to a numpy array
    activity_data = np.array(activity_data)
    # take the mean of the activity data
    activity_data_mean = np.mean(activity_data, axis=0)
    return activity_data_mean

def per_file(filename):
    raw = read_input_data(filename)
    bouts = find_bouts(lids_obj, raw)

    # issue where the length of wakes and sleeps is unequal:
    if bouts == []:
        return (0, 0)
    else:
        # they are equal and all is well.
        activity_data_mean = mean_of_bouts(bouts)    # COMMENT OUT THE INDEX
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
        print('processing file: ', i)
        (file_mean, n_bouts_in_file) = per_file(filenames[i])
        print('file mean: ', file_mean)
        print('n_bouts_in_file: ', n_bouts_in_file)
        if type(file_mean) == int and n_bouts_in_file == 0:    # weird debug to get around file_means issue
            total_bouts += n_bouts_in_file
        else:
            file_means.append(file_mean)  # append the mean of the activity data for each file
            total_bouts += n_bouts_in_file

    # pad arrays shorter than 6 hours with zeros
    padded_file_means = []
    for file in file_means:
        file_length = len(file)
        target_length = 36 # equivalent to 36 * 10 min = 360 minutes = 6 hours
        if file_length < target_length:
            # Calculate the amount of padding needed
            padding = target_length - file_length

            # Pad the array with zeros on the right side only, a constant value
            padded_array = np.pad(file, (0, padding), 'constant')
            padded_file_means.append(padded_array)
        else:
            padded_file_means.append(file)

    plt.figure()

    # Define the x-axis values in minutes, to show up as singular minutes instead of 10-minute increments
    x_axis_minutes = np.arange(0, 360, 10)  # Assuming 36 data points with 10-minute increments

    # set label for x axis
    plt.xlabel('minutes since sleep onset')
    # set label for y axis
    plt.ylabel('inactivity')
    # set title
    n_files = len(padded_file_means)

    # calculate the mean for the plot
    total_mean = np.zeros(36) # 36 is the length of each file. We want to average across all files, manually
    for i in range(len(total_mean)):
        sum = 0
        count = 0
        for fmean in padded_file_means:
            val = fmean[i]
            if val != 0:
                sum += val
                count += 1
        if count != 0:
            total_mean[i] = sum / count

    plt.title(f'mean of {total_bouts} bouts from {n_files} files')
    plt.plot(x_axis_minutes, total_mean)
    print('total mean: ', total_mean)
    plt.show()

def plot_bouts(bouts):
    n_bouts = len(bouts)
    for i in range(n_bouts):
        fig = plt.figure()
        fig.suptitle('bout ' + str(i+1))
        to_plot = np.array(extract_activity_data(bouts[i]).tolist())
        plt.plot(to_plot)
    #plt.show()


def cosine_fit(lids_obj, bout):
    """
    Fit a cosine to the bout data
    :param lids_obj:
    :param bout:
    :return: period
    """
    lids_obj.lids_fit(lids=bout, nan_policy='omit', verbose=False)

    # statistics
    lids_period = lids_obj.lids_fit_results.params['period']
    #print('LIDS period: ', lids_period)

    correlation_factor = lids_obj.lids_pearson_r(bout)
    #print('pearson correlation factor: ', correlation_factor)
    return lids_period.value

def normalize_to_period(activity, period):
    """
    Normalize the activity data to the period
    :param activity: list of activity data
    :param period: period of the activity data
    :return: normalized activity data
    """
    normalized = []
    for i in range(len(activity)):
        normalized.append((i/period, activity[i]))

    n2 = []
    for i in range(len(normalized)):
        n2.append(np.array(normalized[i]))
    n3 = np.array(n2)
    return n3

def find_max_x_value(bouts):
    max_x_value = 0
    for bout in bouts:
        last_x_y = bout[-1]
        print (last_x_y)
        max_x_value = max(max_x_value, last_x_y[0])
    return max_x_value

def find_y_average_in_bin(normalized_bouts_array, bin_x_start, bin_x_end):
    """
    Find the average y value in the bin
    :param normalized_bouts_array: array of normalized bouts
    :param bin_x_start: start of the bin
    :param bin_x_end: end of the bin
    :return: average y value in the bin
    """
    sum_y_values = 0
    count = 0
    for bout in normalized_bouts_array:
        for x_y in bout:
            if x_y[0] >= bin_x_start and x_y[0] < bin_x_end:
                sum_y_values += x_y[1]
                count += 1
    if count == 0:
        return 0
    else:
        return sum_y_values / count

def mean_of_bouts_normalized(lids_obj, bouts):
    """
    Find the mean of the normalized bouts
    :param lids_obj:
    :param bouts:
    :return:
    """
    # extract the activity data from the bouts
    activity_data = []
    periods = []
    for i in range(len(bouts)):  # iterate through the bouts
        activity_data.append(extract_activity_data(bouts[i]))  # append the activity data from each bout
        periods.append(cosine_fit(lids_obj, bouts[i]))

    normalized_bouts_list= []
    for i in range(len(activity_data)):
        normalized_bouts_list.append(normalize_to_period(activity_data[i], periods[i]))

    normalized_bouts_array = np.array(normalized_bouts_list)

    # define a number of bins
    N_BINS = 50 # CHANGE THIS BACK - to 1000 for bins = 30 sec if applicable
    # divide the max X value by the number of bins
    x_increment = max_x_value / N_BINS
    # iterate through the bins and find the average of the Y values in each bin
    x = 0
    y_averages = []
    while x < max_x_value:
        # find the average of the Y values
        bin_x_start = x
        bin_x_end = x + x_increment
        y_average_in_bin = find_y_average_in_bin(normalized_bouts_array,bin_x_start, bin_x_end )
        y_averages.append(y_average_in_bin)
        x += x_increment
    return y_averages

def per_file_normalized(lids_obj, filename):
    # issue where the length of wakes and sleeps is unequal:

    raw = read_input_data(filename)
    bouts = find_bouts(lids_obj, raw)
    if bouts == []:
        return (0, 0)
    else:
        # they are equal and all is well.
        activity_data_mean = mean_of_bouts_normalized(lids_obj, bouts)
        return (activity_data_mean, len(bouts))

def handle_files(filenames):
    file_means = []
    total_bouts = 0
    for i in range(len(filenames)):
        (file_mean, n_bouts_in_file) = per_file(lids_obj, filenames[i])
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

def file_data_plot(file_mean, filename):
    plt.figure()
    # set label for x axis
    plt.xlabel('period')
    # set label for y axis
    plt.ylabel('inactivity')
    # set title
    plt.title(f'file {filename} mean')
    plt.plot(file_mean)

def resample_bouts(sleep_bouts):
    """
    Resamples the bouts to 10 minute intervals
    :param sleep_bouts:  time series of sleep bouts
    :return: time series of sleep bouts resampled to 10 minute intervals
    """
    bouts_resampled = []
    for i in range(len(sleep_bouts)):
        resampled = sleep_bouts[i].resample("10Min").sum()
        bouts_resampled.append(resampled)
    return bouts_resampled

def process_normalized(filenames):
    file_means = []
    total_bouts = 0
    for i in range(len(filenames)):

        print('file being processed: ', i)
        print('associated filename: ', filenames[i])


        (file_mean, n_bouts_in_file) = per_file_normalized(lids_obj, filenames[i])
        if file_mean == 0 and n_bouts_in_file == 0:
            total_bouts += n_bouts_in_file
        else:
            file_means.append(file_mean)  # append the mean of the activity data for each file
            total_bouts += n_bouts_in_file
            file_data_plot(file_mean, filenames[i])
    length_of_shortest_array = find_shortest_series(file_means)
    print("length of shortest array: ", length_of_shortest_array)
    for i in range(len(file_means)):
        file_means[i] = file_means[i][:length_of_shortest_array]
    plt.figure()
    # set label for x axis
    plt.xlabel('period (TBD)')
    # set label for y axis
    plt.ylabel('inactivity')
    # set title
    n_files = len(file_means)
    file_mean = np.mean(file_means, axis=0)
    plt.title(f'mean of {total_bouts} bouts from {n_files} files')

    # set x axis to show LIDS periods
    xs = np.linspace(0, MAX_PERIODS, len(file_mean))

    plt.plot(xs, file_mean)


def LIDS_period_histogram(filenames):
    """
    Create a histogram for distribution of LIDS periods.
    :param LIDS_obj:
    :param filenames:
    :return:
    """

    all_mean_LIDS_periods = []

    for file in filenames:
        # generate raw data
        raw = read_input_data(filename)

        # transform raw into LIDS data
        lids_transformed = lids_obj.lids_transform(ts=raw.data)
        lids_transformed = lids_transformed.resample("10Min").sum() # downscale to 10Min bins
        lids_obj.lids_fit(lids=lids_transformed, nan_policy='omit', verbose=False) # perform the LIDS fit on the data.

        # determine the LIDS period
        lids_period = lids_obj.lids_fit_results.params['period']

        # add to list of all mean LIDS periods from the files
        all_mean_LIDS_periods.append(lids_period)

        # plot a histogram of all the LIDS periods and their distributions
        # Define the bin edges based on the period
        bin_edges = np.arange(min(all_mean_LIDS_periods), max(all_mean_LIDS_periods) + 2) - 0.5

    print('all lids periods: ', all_mean_LIDS_periods)

    # Create the histogram
    plt.hist(all_mean_LIDS_periods, bins=bin_edges, edgecolor='black')

    # Set the labels and title
    plt.xlabel('Period (10-minute intervals)')
    plt.ylabel('Count')
    plt.title('Histogram of Periods')

    # Show the plot
    #plt.show()






''' FUNCTION CALLS '''

filename = '67067_0000000131-timeSeries.csv.gz'
raw = read_input_data(filename)

print('raw type: ', type(raw))
print('raw type data: ', type(raw.data))


# test lids functionality
#LIDS_functionality_test(lids_obj, raw)


#### LIDS GRAPHICAL ANALYSIS ####
directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

filenames = [filename for filename in os.listdir(directory) if filename.endswith('timeSeries.csv.gz')]      # CHANGE THIS BACK - to 'timeSeries.csv.gz'

set_up_plot(filenames[0:10])  # FUNCTION CALL FOR NON-NORMALIZED LIDS GRAPH

#process_normalized(filenames)  # FUNCTION CALL FOR NORMALIZED LIDS GRAPH
plt.show()

## LIDS histogram of period lengths ##
#LIDS_period_histogram(filenames[1:10])











