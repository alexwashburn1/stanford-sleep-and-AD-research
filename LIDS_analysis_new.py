'''IMPORTS'''
import sys
import pyActigraphy
from matplotlib import gridspec
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyActigraphy.analysis import LIDS   #LIDS tools import
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import sex_age_bins_LIDS
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from scipy.stats import sem

# Create a LIDS object
lids_obj = LIDS()

# set max LIDS periods value
MAX_PERIODS = 4
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
    print('about to plot (b)')
    plt.plot(lids_transformed)



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
    for i in range(len(wakes)):
        sleep_wakes.append((sleeps[i], wakes[i]))  # append the tuple of the on and off times
    # iterate through the sleep_wakes and find the duration/timeseries of each sleep. Append to sleep_bouts.
    sleep_bouts = []
    for i in range(len(sleep_wakes)):
        sleep_bouts.append(extract_bout(raw.data, sleep_wakes[i][0], sleep_wakes[i][1])) # sleep_wakes[i]

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


    #TODO - COMMENT OUT the code below if you don't want to delete the first 4 epochs
    all_bouts_first_4_epochs_removed = []
    for bout in bouts_transformed:
        new_bout = bout[0:] # change to 4 if desired to filter out first 4 epochs.
        all_bouts_first_4_epochs_removed.append(new_bout)

    return all_bouts_first_4_epochs_removed


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


def per_file_no_mean(filename):
    """
    process each file and do not return one mean bout, but a list of all of the bouts
    :param filename: the file to generate a list of bouts for.
    :return: a list of all of the bouts in the file, and the number of bouts in the file.
    """

    raw = read_input_data(filename)
    bouts = find_bouts(lids_obj, raw)

    return (bouts, len(bouts))



def set_up_plot(filenames):
    """
    Takes the mean over several bouts for several files. Filters all files according to the shortest record. Plots.
    Note: all bouts are trimmed to be 6 hours. If there are shorter, they are padded with zeros.
    :param filenames: the filenames to include in the ultimate LIDS graph
    :return: padded_bouts: the bouts trimmed or padded appropriately to be 6 hours.
    """
    total_bouts = 0
    count_of_invalid_bouts = 0
    all_bouts_all_files = []
    for i in range(len(filenames)):
        print('processing file: ', i)
        (all_bouts_from_file, n_bouts_in_file) = per_file_no_mean(filenames[i])

        for j in range(len(all_bouts_from_file)):
            # append the bouts from each file to the total list of bouts
            bout_with_filename = (all_bouts_from_file[j], filenames[i])

            (lids_period, r, p, MRI, onset_phase, offset_phase) = cosine_fit(lids_obj, bout_with_filename[0])

            if p < 0.05 and str(lids_period) != '17.5': #(can comment back in)
                all_bouts_all_files.append(bout_with_filename)
            else:
                print('lids period: ', lids_period)
                count_of_invalid_bouts += 1

        total_bouts += n_bouts_in_file

    # Iterate over all bouts, removing the first 4 epochs. TODO - DELETE IF NOT WANTED
    temp_all_bouts = []
    for bout in all_bouts_all_files:
        filename = bout[1]
        new_bout = bout[0].iloc[0:] # change to 4 if desired to filter out first 4 epochs.
        bout_tuple = (new_bout, filename)

        temp_all_bouts.append(bout_tuple)

    # pad arrays shorter than 6 hours with zeros
    padded_bouts = []
    for bout in temp_all_bouts:
        filename = bout[1]
        bout_length = len(bout[0]) # bout[0] should represent the bout itself
        target_length = 36
        if bout_length < target_length:
            # Calculate the amount of padding needed
            padding = target_length - bout_length

            # Pad the array with zeros on the right side only, a constant value
            padded_array = np.pad(bout[0], (0, padding), 'constant')
            bout_and_filename = (padded_array, filename)
            padded_bouts.append(bout_and_filename)
        else:
            bout_and_filename = (bout[0][0:36], filename)
            padded_bouts.append(bout_and_filename)


    plt.figure()

    # Define the x-axis values in minutes, to show up as singular minutes instead of 10-minute increments
    x_axis_minutes = np.arange(0, 360, 10)

    # set label for x axis
    plt.xlabel('minutes since sleep onset')
    # set label for y axis
    plt.ylabel('inactivity')

    # calculate the mean of all the bouts, for the plot
    total_mean = np.zeros(36)
    for i in range(len(total_mean)):
        sum = 0
        count = 0
        for bout in padded_bouts:
            val = bout[0][i]
            if val != 0:
                sum += val
                count += 1
        if count != 0:
            total_mean[i] = sum / count


    # confidence interval attempt:

    confidence_intervals = []
    for i in range(len(total_mean)):
       print('i: ', i)
       bout_values = [bout[0][i] for bout in padded_bouts if bout[0][i] != 0]
       confidence_intervals.append(1.96 * sem(bout_values))  # 95% confidence interval, assuming a normal distribution

    plt.title(f'mean of {len(padded_bouts)} bouts')

    print('about to plot (c)')
    plt.plot(x_axis_minutes, total_mean)
    plt.fill_between(x_axis_minutes, total_mean - confidence_intervals, total_mean + confidence_intervals,
                     alpha=0.2)


    return padded_bouts

def box_plot_outliers(padded_bouts):
    """
    Plot 5 box plots for the first 50 minutes of sleep onset from the bouts, to determine outliers.
    :param padded_bouts:
    :return:
    """
    # Find the number of bouts and the length of each bout
    bout_length = len(padded_bouts[0][0])  # Since all bouts have the same length

    # extract only the bouts, not the filenames for the following processes
    extracted_bouts = []
    for bout in padded_bouts:
        extracted_bouts.append(bout[0]) # bout[0] is the bout, not the filename
        print('filename: ', bout[1])

    # Convert padded_bouts to a 2D numpy array
    padded_bouts_array = np.array(extracted_bouts)

    # Extract the first five values from each bout
    first_five_values = padded_bouts_array[:, :5] # syntax: get the first 5 values for ALL rows in padded_bouts_array

    # Get the indices of the 3 highest outliers for epoch #1 (first column)
    outliers_indices = np.argsort(first_five_values[:, 0])[-3:]

    # Get the filename values of the 3 highest outliers for epoch # 1
    outlier_index_1 = outliers_indices[0]
    outlier_index_2 = outliers_indices[1]
    outlier_index_3 = outliers_indices[2]
    filename_1 = padded_bouts[outlier_index_1][1]
    filename_2 = padded_bouts[outlier_index_2][1]
    filename_3 = padded_bouts[outlier_index_3][1]
    outlier_filenames = [filename_1, filename_2, filename_3]
    print('filenames of the outliers: ', outlier_filenames)
    # filenames of the outliers:  ['68307_0003900477-timeSeries.csv.gz', '75189_0000001011-timeSeries.csv.gz', '67180_0003900497-timeSeries.csv.gz']

    # print the bout timing so I can see which specific one it is in the filename
    print('bout right before graph # 1 bout: ', padded_bouts[outlier_index_1 - 1][0])
    print('bout from graph #1: ', padded_bouts[outlier_index_1][0])
    print('bout right after graph # 1 bout: ', padded_bouts[outlier_index_1 + 1][0])
    print('bout from graph #2: ', padded_bouts[outlier_index_2][0])
    print('bout from graph #3: ', padded_bouts[outlier_index_3][0])

    # Create the box plot
    plt.violinplot(first_five_values)

    # Set labels and title
    plt.xlabel('10 minute interval')
    plt.ylabel('Inactivity')
    plt.title('Violin Plot of First Five Values from Sleep Bouts')

    # Scatter plot for ALL data points. Iterates over each bout in padded_bouts and plots first 5 epochs.
    # Jitter magnitude (you can adjust this value to control the amount of jitter)
    x_jitter_magnitude = 0.1
    y_jitter_magnitude = 0.1

    for bout_data in padded_bouts:
        # generate jittered x-axis values
        x_jittered = [x_val + np.random.uniform(-x_jitter_magnitude, x_jitter_magnitude) for x_val in range(1, 6)]

        # Generate jittered y-axis values
        y_jittered = [y_val + np.random.uniform(-y_jitter_magnitude, y_jitter_magnitude) for y_val in bout_data[0][:5]]

        # Plot the original x-axis values with jittered y-axis values
        plt.scatter(x_jittered, y_jittered, marker='o', alpha=0.3, color='blue', s=6)


    print('outliers indices: ', outliers_indices)
    return outliers_indices




def plot_outlier_bouts(outliers_indices, padded_bouts):
    """
    Plots the LIDS transformed bouts for the the 3 highest outliers for the first epoch (10 minute interval)
    :param outliers_indices: the indices of the outlier bouts
    :param padded_bouts: all the bouts (to be indexed with the outlier indices)
    :return:
    """
    # outlier indices: 1 = 1213, 2 = 1163, 3 = 520 (for the entire data set)
    # extract individual bouts
    bout_1 = padded_bouts[1213]
    print('length bout 1: ', len(bout_1))
    bout_2 = padded_bouts[1163]
    print('length bout 2: ', len(bout_2))
    bout_3 = padded_bouts[520]
    print('length bout 3: ', len(bout_3))

    bouts_data = [bout_1, bout_2, bout_3]
    n_bouts = len(bouts_data)
    colors = ['red', 'blue', 'green']

    # Create individual subplots for each bout
    fig, axes = plt.subplots(n_bouts, 1, figsize=(8, 6))

    # define x axis minutes for even x axis
    x_axis_minutes = np.arange(0, 360, 10)

    # Loop over each subplot and plot the inactivity data for each bout
    for i, (ax, bout) in enumerate(zip(axes, bouts_data)):
        ax.plot(x_axis_minutes, bout, color=colors[i])
        ax.set_title(f'Bout {i + 1}')
        ax.set_ylabel('Inactivity')
        ax.set_ylim(0, max(max(bout) for bout in bouts_data))  # Set y-axis limits for all subplots

    axes[-1].set_xlabel('Time since sleep onset (minutes)')  # Set x-axis label for the last subplot

    # Adjust layout to avoid overlap of subplots
    plt.tight_layout()




def cosine_fit(lids_obj, bout):
    """
    Fit a cosine to the bout data
    :param lids_obj:
    :param bout:
    :return: period
    """
    lids_obj.lids_fit(lids=bout, nan_policy='omit', bounds=('30min', '180min'), verbose=False) # CHANGE TO 180 MIN

    # statistics
    lids_period = lids_obj.lids_fit_results.params['period'].value # comment in * 10

    if str(lids_period) == '17.5':
        x_axis_minutes = np.arange(0, len(bout) * 10, 10)
        temp_bout = bout
        #plt.figure()
        #plt.xlabel('minutes since sleep onset')
        #plt.ylabel('inactivity')
        #plt.title('LIDS transformed data for bout with LIDS period >= 175')
        #plt.plot(x_axis_minutes, temp_bout)
        #plt.show()


    else:
        x_axis_minutes = np.arange(0, len(bout) * 10, 10)
        temp_bout = bout
        #plt.figure()
        #plt.xlabel('minutes since sleep onset')
        #plt.ylabel('inactivity')
        #plt.title(f'n=1 sleep bout')
        #plt.plot(x_axis_minutes, temp_bout)
        #plt.show()



    (r, p) = lids_obj.lids_pearson_r(bout)

    MRI = lids_obj.lids_mri(bout)

    (onset_phase, offset_phase) = lids_obj.lids_phases(bout)

    # create a dataframe for all bouts with LIDS period, r, p, MRI, onset phase, and offset phase as columns
    return (lids_period, r, p, MRI, onset_phase, offset_phase)



def normalize_to_period(activity, period, onset_phase):
    """
    Normalize the activity data to the period
    :param activity: list of activity data
    :param period: period of the activity data
    :return: normalized activity data
    """
    normalized = []
    onset_phase = onset_phase / 360 # convert from degrees to units of period. Onset phase is the "black dot" on Figure 5c Winnebeck 2018.
    for i in range(len(activity)):
        normalized_x = i / period
        x_shifted = normalized_x - onset_phase
        if x_shifted >= 0:
            #it will only be >0 past the black dot (Fig 5c), doing the "cutting" as referenced in the methods of Winnebeck 2018.
            normalized.append((x_shifted, activity[i]))

    # convert the list to a numpy array
    normalized2 = []
    for i in range(len(normalized)):
        normalized2.append(np.array(normalized[i])) # make an array of the tuples
    normalized3 = np.array(normalized2)
    return normalized3

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
    bout_values = []
    count = 0
    for bout in normalized_bouts_array:
        for x_y in bout:
            if x_y[0] >= bin_x_start and x_y[0] < bin_x_end:
                sum_y_values += x_y[1]
                count += 1
                bout_values.append(x_y[1])
    if count == 0:
        average = 0
    else:
        average = sum_y_values / count

    confidence_interval = 1.96 * sem(bout_values)  # 95% confidence interval, assuming a normal distribution
    return (average, confidence_interval)

def hist_of_periods(periods):
    #### HISTOGRAM OF PERIODS ####
    # get the value of periods in minutes for the histogram of period distribution
    periods_for_hist = [period * 10 for period in periods]

    # Calculate the minimum and maximum values and cast them to integers
    min_value = int(min(periods_for_hist))
    max_value = int(max(periods_for_hist))

    # Set the bin edges using a range with intervals of 10
    bins = range(min_value, max_value + 10, 10)

    # Create a 2x1 grid of subplots with shared x-axis
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # Plot the histogram with clearly outlined bins and y-axis labels
    ax1.set_ylabel('Counts')
    ax1.hist(periods_for_hist, bins=bins, edgecolor='black')

    # Customize the box plot appearance
    box_plot = ax2.boxplot(periods_for_hist, vert=False, positions=[min(periods_for_hist) + 5], widths=6,
                           patch_artist=True)
    for box in box_plot['boxes']:
        box.set_facecolor('white')  # Set box color to non-transparent white

    ax2.yaxis.set_visible(False)
    ax2.set_xticks([])  # Hide x-axis labels for the box plot

    # Set the same x-axis limits for both axes to ensure alignment
    ax1.set_xlim(ax1.get_xlim())  # Copy x-axis limits from the main histogram
    ax2.set_xlim(ax1.get_xlim())  # Set the same limits for the box plot

    plt.xticks(bins)  # Set x-axis ticks for the histogram

    plt.xlabel('Period (min)')

    # Add a text box in the top-right corner displaying the number of periods
    n_periods = len(periods_for_hist)
    text_box = f'n = {n_periods} bouts'
    ax1.text(0.03, 0.95, text_box, transform=ax1.transAxes, ha='left', va='top',
             bbox=dict(facecolor='white', edgecolor='black'))

    #plt.show()


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
    onset_phases = []

    # init empty dataframe for all bouts with LIDS period, r, p, MRI, onset phase, and offset phase as columns.
    bout_info_df = pd.DataFrame(columns=['LIDS period', 'r', 'p', 'MRI', 'onset phase', 'offset phase'])

    for i in range(len(bouts)):  # iterate through the bouts

        (lids_period, r, p, MRI, onset_phase, offset_phase) = cosine_fit(lids_obj, bouts[i])

        # append a row to the dictionary for the bout in question
        row_data = {
            'LIDS period': lids_period,
            'r': r,
            'p': p,
            'MRI': MRI,
            'onset phase': onset_phase,
            'offset phase': offset_phase
        }
        bout_info_df = bout_info_df.append(row_data, ignore_index=True)


        if p < 0.05 and str(lids_period) != '17.5':
            periods.append(lids_period)
            onset_phases.append(onset_phase)
            activity_data.append(extract_activity_data(bouts[i]))  # append the activity data from each bout


    # export bout info df to csv:
    filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/bout_summary_metrics_csv/'
    #bout_info_df.to_csv(filepath + 'bout_summary_statistics.csv')

    # normalize the bouts

    normalized_bouts_list= []
    for i in range(len(activity_data)):
        normalized_bouts_list.append(normalize_to_period(activity_data[i], periods[i], onset_phases[i]))

    normalized_bouts_array = np.array(normalized_bouts_list)

    # plot a histogram of the periods
    #hist_of_periods(periods)

    # define a number of bins
    N_BINS = 50 # CHANGE THIS BACK to 50 - to 1000 for bins = 30 sec if applicable
    # divide the max X value by the number of bins
    x_increment = max_x_value / N_BINS
    # iterate through the bins and find the average of the Y values in each bin
    x = 0
    y_averages = []
    confidence_intervals = []
    while x < max_x_value:
        # find the average of the Y values
        bin_x_start = x
        bin_x_end = x + x_increment
        (y_average_in_bin, confidence_interval) = find_y_average_in_bin(normalized_bouts_array,bin_x_start, bin_x_end )
        y_averages.append(y_average_in_bin)
        confidence_intervals.append(confidence_interval)
        x += x_increment
    return (y_averages, confidence_intervals, len(periods))

def file_data_plot(file_mean, filename):
    plt.figure()
    # set label for x axis
    plt.xlabel('period')
    # set label for y axis
    plt.ylabel('inactivity')
    # set title
    plt.title(f'file {filename} mean')
    print('about to plot (d)')
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
    """

    :param filenames:
    :param label: label, for legend if applicable.
    :param colors: a list of colors, if using bins will have multiple colors.
    :return:
    """
    all_bouts = []
    total_bouts = 0
    n_files = 0
    for i in range(len(filenames)):

        print('file being processed: ', i)
        print('associated filename: ', filenames[i])


        (all_bouts_from_file, n_bouts_in_file) = per_file_no_mean(filenames[i])

        # append each bout from the file seperately to the list of ALL bouts
        for bout in all_bouts_from_file:
            all_bouts.append(bout)

        total_bouts += n_bouts_in_file
        n_files += 1

    (activity_mean, confidence_intervals, num_bouts) = mean_of_bouts_normalized(lids_obj, all_bouts)
    print('length activity_mean: ', len(activity_mean))

    # set x axis to show LIDS periods
    xs = np.linspace(0, MAX_PERIODS, len(activity_mean))

    # Smooth the line a little bit
    x_smooth = np.linspace(min(xs), max(xs), 300)  # Increase 300 to get a smoother line
    spl = make_interp_spline(xs, activity_mean)
    file_mean_smooth = spl(x_smooth)

    #age_interval_index = int(label.split('-')[0].split()[-1])  # Extract age interval from label
    # Ensure that the x-axis ticks show integers (0, 1, 2, 3, 4, ...)
    num_ticks = MAX_PERIODS + 1 # You can adjust the number of ticks as needed
    plt.xticks(np.linspace(0, MAX_PERIODS, num_ticks), np.arange(num_ticks))

    print('about to plot (e)')
    plt.xlabel('LIDS cycle')
    plt.ylabel(f'Normalized Activity from {num_bouts} bouts')
    plt.plot(x_smooth, file_mean_smooth)

def process_normalized_with_confidence_intervals(filenames, label, colors):
    """

        :param filenames:
        :param label: label, for legend if applicable.
        :param colors: a list of colors, if using bins will have multiple colors.
        :return:
        """
    all_bouts = []
    total_bouts = 0
    n_files = 0
    for i in range(len(filenames)):

        print('file being processed: ', i)
        print('associated filename: ', filenames[i])

        (all_bouts_from_file, n_bouts_in_file) = per_file_no_mean(filenames[i])

        # append each bout from the file seperately to the list of ALL bouts
        for bout in all_bouts_from_file:
            all_bouts.append(bout)

        total_bouts += n_bouts_in_file
        n_files += 1

    (activity_mean, confidence_intervals, num_bouts) = mean_of_bouts_normalized(lids_obj, all_bouts)
    for val in activity_mean:
        print('val from activity mean: ', val)

    # set x axis to show LIDS periods
    # shave activity mean to be of length 36, in order to be consistent with the len of bouts.
    xs = np.linspace(0, MAX_PERIODS, len(activity_mean))

    # Smooth the line a little bit
    x_smooth = xs
    spl = make_interp_spline(xs, activity_mean)
    file_mean_smooth = spl(x_smooth)

    # age_interval_index = int(label.split('-')[0].split()[-1])  # Extract age interval from label
    if colors != '':
        color = colors[label]  # Normalize age_interval_index to [0, 1]

    # Ensure that the x-axis ticks show integers (0, 1, 2, 3, 4, ...)
    num_ticks = 5  # You can adjust the number of ticks as needed
    plt.xticks(np.linspace(0, MAX_PERIODS, num_ticks), np.arange(num_ticks))

    # Calculate coefficients for the linear line of best fit
    coefficients = np.polyfit(x_smooth, file_mean_smooth, deg=1)

    # Generate the y values for the line of best fit
    line_of_best_fit = np.polyval(coefficients, x_smooth)

    # CASE FOR NON-BINNED DATA.
    if colors == '' and label == '':
        print('length file mean smooth: ', len(file_mean_smooth))
        for val in file_mean_smooth:
            print('val: ', val)
        plt.xlabel('LIDS cycle')
        plt.ylabel('Inactivity')
        plt.title(f'Normalized Activity from {num_bouts} bouts')

        plt.plot(x_smooth, file_mean_smooth)
        plt.fill_between(x_smooth, file_mean_smooth - confidence_intervals, file_mean_smooth + confidence_intervals,
                         alpha=0.2)

    else:
        # CASE FOR BINNED DATA - setting up plot already done in helper methods.
        # Plot the line of best fit using the same color as the one passed as a parameter
        print('about to plot (f)')
        plt.plot(x_smooth, line_of_best_fit, color=color, linestyle='dashed')

        plt.plot(x_smooth, file_mean_smooth, label=label, color=color)
        plt.fill_between(x_smooth, file_mean_smooth - confidence_intervals, file_mean_smooth + confidence_intervals,
                     color=color, alpha=0.2)


def normalized_binned_by_sex(filenames, age_sex_etiology_dict):
    """
    Generates a list of filenames for subjects of sex Male and Female, by looking them up in the demographics
    dictionary.
    :param filenames: the list of all 166 filenames
    :param age_sex_etiology_dict: the dictionary containing sex information about all n=166 filenames
    :return: male_female_file_list_tuple - a tuple, containing the male and female filename lists, respectively.
    """
    male_filenames_list = []
    female_filenames_list = []

    for filename in filenames:
        filename = filename.replace("-timeSeries.csv.gz", "")
        sex = age_sex_etiology_dict[filename][1]
        if sex == 'Male':
            male_filenames_list.append(filename + "-timeSeries.csv.gz")
        else:
            # sex must be Female
            female_filenames_list.append(filename + "-timeSeries.csv.gz")

    male_female_file_list_tuple = (male_filenames_list, female_filenames_list)
    return male_female_file_list_tuple

def normalized_binned_by_age(filenames, age_sex_etiology_dict, bin_type):
    """

    :param filenames:
    :param age_sex_etiology_dict:
    :param bin_type: describes which way to bin by age, when trying different binning iterations
    :return:
    """
    if bin_type == 'Normal':
        age_40to70_list = []
        age_70to80_list = []
        age_80to100_list = []

        for filename in filenames:
            filename = filename.replace("-timeSeries.csv.gz", "")
            age = age_sex_etiology_dict[filename][0]
            if age >= 40 and age < 70:
                age_40to70_list.append(filename + "-timeSeries.csv.gz")
            elif age >= 70 and age < 80:
                age_70to80_list.append(filename + "-timeSeries.csv.gz")
            elif age >= 80 and age <= 100:
                age_80to100_list.append(filename + "-timeSeries.csv.gz")

        print('len filename 40-70: ', len(age_40to70_list))
        print('len filename 70-80: ', len(age_70to80_list))
        print('len filename 80-100: ', len(age_80to100_list))

        return (age_40to70_list, age_70to80_list, age_80to100_list)

    elif bin_type == 'over_under_75':
        under_75_files = []
        over_75_files = []

        for filename in filenames:
            filename = filename.replace("-timeSeries.csv.gz", "")
            age = age_sex_etiology_dict[filename][0]
            if age < 75:
                under_75_files.append(filename + "-timeSeries.csv.gz")
            else:
                # age is over 75
                over_75_files.append(filename + "-timeSeries.csv.gz")

        print('files under 75: ', len(under_75_files))
        print('files over 75: ', len(over_75_files))
        return (under_75_files, over_75_files)

def normalized_binned_by_etiology(filenames, age_sex_etiology_dict):
    """
    Generates a list of filenames for subjects of sex Male and Female, by looking them up in the demographics
    dictionary.
    :param filenames: the list of all 166 filenames
    :param age_sex_etiology_dict: the dictionary containing sex information about all n=166 filenames
    :return: male_female_file_list_tuple - a tuple, containing the male and female filename lists, respectively.
    """
    HC_filenames_list = []
    AD_filenames_list = []
    LB_filenames_list = []
    other_filenames_list = []   # ignore this, since we won't plot them.

    for filename in filenames:
        filename = filename.replace("-timeSeries.csv.gz", "")
        etiology = age_sex_etiology_dict[filename][2]
        if etiology == 'HC':
            HC_filenames_list.append(filename + "-timeSeries.csv.gz")
        elif etiology == 'AD':
            AD_filenames_list.append(filename + "-timeSeries.csv.gz")
        elif etiology == 'LB':
            LB_filenames_list.append(filename + "-timeSeries.csv.gz")
        else:
            # etiology must be "other"
            other_filenames_list.append(other_filenames_list)

    etiology_file_list_tuple = (HC_filenames_list, AD_filenames_list, LB_filenames_list)
    return etiology_file_list_tuple     # order of tuple is HC, AD, LB

def set_up_plot_sex_binned(filenames, age_sex_etiology_dict):

    (male_filenames, female_filenames) = normalized_binned_by_sex(filenames, age_sex_etiology_dict)
    # Define the age intervals and lists
    sex_intervals = ['Male', 'Female']
    sex_lists = [male_filenames, female_filenames]
    # create a color map
    # Define custom colors for the colormap
    colors_sex = {
        'Male': (0, 0, 1),  # Blue
        'Female': (1, 0.5, 0.5) # Pink/orange
    }
    # Create the figure
    plt.xlabel('LIDS cycle')
    plt.ylabel('Inactivity')
    plt.title('Normalized Activity')
    # Call process_normalized for each age interval and plot with appropriate color
    for i, sex in enumerate(sex_intervals):
        process_normalized_with_confidence_intervals(sex_lists[i], sex, colors_sex)
    # Add the legend with custom title and location
    plt.legend(title='Sex', loc='upper right')


def set_up_plot_age_binned_normal(filenames, age_sex_etiology_dict):
    (age_40to70_list, age_70to80_list, age_80to100_list) = normalized_binned_by_age(filenames, age_sex_etiology_dict, 'Normal') # bin type is 'Normal'
    # Define the age intervals and lists
    age_intervals = ['age 40-70', 'age 70-80', 'age 80-100']
    age_lists = [age_40to70_list, age_70to80_list, age_80to100_list]
    # create a color map
    # Define custom colors for the colormap
    colors_age = {
        'age 40-70': (0, 0.7, 0),  # Green (RGB values from 0 to 1)
        'age 70-80': (0.4, 0.4, 0.9),  # Blue
        'age 80-100': (0.7, 0, 0.7),  # Red
    }
    # Create the figure
    plt.xlabel('LIDS cycle')
    plt.ylabel('Inactivity')
    plt.title('Normalized Activity')
    # Call process_normalized for each age interval and plot with appropriate color
    for i, age_interval in enumerate(age_intervals):
        process_normalized_with_confidence_intervals(age_lists[i], age_interval, colors_age)
    # Add the legend with custom title and location
    plt.legend(title='Age Interval', loc='upper right')


def set_up_plot_age_binned_over_under_75(filenames, age_sex_etiology_dict):
    (under_75_list, over_75_list) = normalized_binned_by_age(filenames, age_sex_etiology_dict, 'over_under_75')
    # Define the age intervals and lists
    age_intervals = ['under 75', 'over 75']
    age_lists = [under_75_list, over_75_list]
    # create a color map
    # Define custom colors for the colormap
    colors_age = {
        'under 75': (0, 1, 0),  # Green (RGB values from 0 to 1)
        'over 75': (1, 0, 0),  # Red
    }
    # Create the figure
    plt.xlabel('LIDS cycle')
    plt.ylabel('Inactivity')
    plt.title('Normalized Activity')
    # Call process_normalized for each age interval and plot with appropriate color
    for i, age_interval in enumerate(age_intervals):
        process_normalized_with_confidence_intervals(age_lists[i], age_interval, colors_age)
    # Add the legend with custom title and location
    plt.legend(title='Age Interval', loc='upper right')


def set_up_plot_binned_etiology(filenames, age_sex_etiology_dict):
    (HC_filenames_list, AD_filenames_list, LB_filenames_list) = normalized_binned_by_etiology(filenames, age_sex_etiology_dict)
    # Define the age intervals and lists
    etiology_intervals = ['HC', 'AD', 'LB']
    etiology_lists = [HC_filenames_list, AD_filenames_list, LB_filenames_list]
    # create a color map
    # Define custom colors for the colormap
    colors_etiology = {
        'HC': (0, 1, 0),  # Green (RGB values from 0 to 1)
        'AD': (1, 0, 0),  # Red
        'LB': (0, 0, 1)   # Blue
    }
    # Create the figure
    plt.xlabel('LIDS cycle')
    plt.ylabel('Inactivity')
    plt.title('Normalized Activity')
    # Call process_normalized for each age interval and plot with appropriate color
    for i, etiology in enumerate(etiology_intervals):
        process_normalized_with_confidence_intervals(etiology_lists[i], etiology, colors_etiology)
    # Add the legend with custom title and location
    plt.legend(title='Etiology', loc='upper right')

def visualize_roenneberg_sleep_bouts_transormed(filename):
    """
    Visualize how Roenneberg detects sleep bouts on raw actigraphy data. Then, run the LIDS transform on a SINGLE bout
    within the actigraphy record to visualize the transform (Mirrors Winnebeck fig 1B)
    :param filename: the filename corresponding to the subject's Actigraphy record to visualize
    :return:
    """
    # retrieve the raw data
    raw = read_input_data(filename)

    # Replace values greater than 400 with 150
    raw.data.loc[raw.data > 400] = 150

    # layout
    layout = go.Layout(
        title="Rest/Activity detection",
        xaxis=dict(title="Date time"),
        yaxis=dict(showticklabels=False), # for raw data
        yaxis2=dict(overlaying='y', side='right', showticklabels=False), # for Roenneberg
        yaxis3={"overlaying": "y", "side": "right", "showticklabels": False, "range": [11.5, 110]}, # for LIDS transformed data
        showlegend=True,
        plot_bgcolor='white',  # Set background color to white
        xaxis_showgrid=False,  # Hide x-axis grid lines
        yaxis_showgrid=False  # Hide y-axis grid lines
       # yaxis2_showgrid=False,  # Hide y-axis2 grid lines
       # yaxis3_showgrid=False  # Hide y-axis3 grid lines
    )

    # retrieve onset and offset times for the sleep bouts
    (wakes, sleeps) = raw.Roenneberg_AoT()
    i = 5

    # retrieve the bout in question (raw data)
    bout = extract_bout(raw.data, sleeps[i], wakes[i])

    # retrieve the narrowed-down record of data
    raw_sleep_data = extract_bout(raw.data, wakes[4], sleeps[6])

    raw_sleep_data.loc[raw_sleep_data > 200] = 50 # fix for invalid activity readings

    # create a roenneberg object
    roenneberg = raw.Roenneberg()
    # trim it in accordance with the raw data, for plotting
    trimmed_roenneberg = extract_bout(roenneberg, wakes[4], sleeps[6])

    # LIDS transform the bout in question
    lids_transformed = lids_obj.lids_transform(ts=bout)

    # plot a timeseries of roenneberg sleep detection, raw data, and LIDS transformed data (for one bout)
    roenneberg_fig = go.Figure(data=[
        go.Scatter(x=raw_sleep_data.index.astype(str), y=raw_sleep_data, fill='tozeroy', name='Data'),
        go.Scatter(x=trimmed_roenneberg.index.astype(str), y=trimmed_roenneberg, yaxis='y2', fill='tozeroy', name='Roenneberg', fillcolor='rgba(255, 165, 0, 0.0)'),
        go.Scatter(x=lids_transformed.index.astype(str), y=lids_transformed, yaxis='y3', fill='tozeroy', name='LIDS')
    ], layout=layout)

    roenneberg_fig.show()




''' FUNCTION CALLS '''

#### LIDS GRAPHICAL ANALYSIS ####
directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

filenames = [filename for filename in os.listdir(directory) if filename.endswith('timeSeries.csv.gz')]
print('length filenames: ', len(filenames))
# CHANGE THIS BACK - to 'timeSeries.csv.gz'

# 1) for mean, non-normalized plot
#padded_bouts = set_up_plot(filenames) # FUNCTION CALL FOR NON-NORMALIZED LIDS GRAPH
#plt.show()

# 2) outlier analysis
#outlier_indices = box_plot_outliers(padded_bouts)
#plot_outlier_bouts(outlier_indices, padded_bouts)
#plt.show()

# 3) for the normalized plot, all filenames
#process_normalized_with_confidence_intervals(filenames, '', '')  # FUNCTION CALL FOR NORMALIZED LIDS GRAPH
#plt.show()

### define the dictionary, to look up age, sex, etiology information for each user ###
#age_sex_etiology_dict = sex_age_bins_LIDS.initialize_user_dictionary('AgeSexDx_n166_2023-07-13.csv')

### 4) create the normalized plot, BINNED BY SEX ###
#plt.figure()
#set_up_plot_sex_binned(filenames, age_sex_etiology_dict)

### 5) create the normalized plot, BINNED BY AGE NORMALLY ###
#plt.figure()
#set_up_plot_age_binned_normal(filenames, age_sex_etiology_dict)

### 6) create the normalized plot, BINNED BY AGE OVER UNDER 75 ###
#plt.figure()
#set_up_plot_age_binned_over_under_75(filenames, age_sex_etiology_dict)

### 7) create the normalized plot, BINNED BY ETIOLOGY ###
#plt.figure()
#set_up_plot_binned_etiology(filenames, age_sex_etiology_dict)

##

### 8) create visualization plot of raw data vs LIDS transformed data
#visualize_roenneberg_sleep_bouts_transormed(str(filenames[5]))



#plt.show()

















