'''IMPORTS'''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
from pyActigraphy.analysis import Cosinor  # for cosinor analysis
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress


def prepare_CRAN_data(filepath, CRAN_filename):
    """
     Prepares the CRAN csv data to be compared to the CRAN data, namely by altering column names and sorting.
    :param filepath: the filepath to the CRAN csv file.
    :param CRAN_filename: the filename of the CRAN file (.csv)
    :return: CRAN_metrics_df: a dataframe with the altered CRAN data.
    """
    # read in summary metrics from CRAN
    CRAN_metrics_df = pd.read_csv(filepath + CRAN_filename)

    # Remove the suffix from the filename column to get only the ID
    CRAN_metrics_df["filename"] = CRAN_metrics_df["filename"].str.replace(".cwa$", "")

    # Sort the filename column in CRAN_metrics_df in ascending order
    CRAN_metrics_df = CRAN_metrics_df.sort_values("filename", ascending=True)
    print('CRAN: ', CRAN_metrics_df.head())

    return CRAN_metrics_df


def prepare_pyActigraphy_data(filepath, pyActigraphy_filename):
    """
    Prepares the pyActigraphy csv data to be compared to the CRAN data, namely by altering column names and sorting.
    :param filepath: the filepath to the pyActigraphy csv file.
    :param pyActigraphy_filename: the filename of the pyActigraphy file (.csv)
    :return: pyActigraphy_metrics_df: a dataframe with the altered pyActigraphy data.
    """
    # read in summary metrics from pyActigraphy
    pyActigraphy_metrics_df = pd.read_csv(filepath + pyActigraphy_filename)

    # Remove the suffix from the filename column to get only the ID
    pyActigraphy_metrics_df["filename"] = pyActigraphy_metrics_df["filename"].str.replace("-timeSeries.csv.gz$", "")

    # Sort the "filename" column in pyActigraphy_metrics_df in ascending order.
    pyActigraphy_metrics_df = pyActigraphy_metrics_df.sort_values("filename", ascending=True)
    print('pyActigraphy: ', pyActigraphy_metrics_df.head())

    return pyActigraphy_metrics_df


def CRAN_vs_pyActigraphy_metric_bias(CRAN_df, pyActigraphy_df, metric_to_compare, output_filepath):
    """
    Compare a metric between CRAN and pyActigraphy packages.

    :param CRAN_df: CRAN dataframe.
    :param pyActigraphy_df: pyActigraphy dataframe.
    :param metric_to_compare: the metric to compare between the two packages.
    :param output_filepath: the filepath to save the figure.
    :return:
    """

    # Extract the metric values from the merged DataFrame
    x_values = pyActigraphy_df[metric_to_compare]
    y_values = CRAN_df[metric_to_compare]

    # Create empty lists to store filtered values
    x_filtered = []
    y_filtered = []

    # Iterate over the values and filter points outside the limits - DEBUGGING invalid values for Amplitude
    for x, y in zip(x_values, y_values):
        if x >= 0.05 and y >= 0.05:
            x_filtered.append(x)
            y_filtered.append(y)

    # Convert filtered lists to numpy arrays
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    # Perform linear regression only on filtered values
    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

    # Create a scatter plot with filtered values
    plt.scatter(x_filtered, y_filtered)

    # Set axis labels and title
    plt.xlabel('pyActigraphy ' + metric_to_compare)
    plt.ylabel('CRAN ' + metric_to_compare)
    plt.title('Comparison of pyActigraphy vs CRAN for {}'.format(metric_to_compare))

    # Add a line of best fit (dotted) using filtered values
    line = slope * x_filtered + intercept
    plt.plot(x_filtered, line, 'r--')

    # Calculate R-squared only with filtered values
    r_squared = r_value ** 2

    # add a legend with R^2 and slope
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.4f}\nSlope: {slope:.4f}', ha='left', va='top',
             transform=plt.gcf().transFigure)

    # Determine the minimum and maximum values with a margin
    margin = 0.001  # Adjust the margin as needed
    min_value = min(min(x_values), min(y_values))
    max_value = max(max(x_values), max(y_values))
    xlim = (min_value * (1 - margin), max_value * (1 + margin))
    ylim = (min_value * (1 - margin), max_value * (1 + margin))

    # Set the limits for both axes with the margin
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save the figure to the specified filepath
    plt.savefig(output_filepath)

    # Show the plot
    plt.show()



def CRAN_vs_pyActigraphy_two_metrics(CRAN_df, pyActigraphy_df, pyActigraphy_metric, CRAN_metric, output_filepath):
    """
    Compare a metric between CRAN and pyActigraphy packages.

    :param CRAN_df: CRAN dataframe.
    :param pyActigraphy_df: pyActigraphy dataframe.
    :param metric_to_compare: the metric to compare between the two packages.
    :param output_filepath: the filepath to save the figure.
    :return:
    """

    # Extract the metric values from the merged DataFrame
    x_values = pyActigraphy_df[pyActigraphy_metric]
    y_values = CRAN_df[CRAN_metric]

    # Create empty lists to store filtered values
    x_filtered = []
    y_filtered = []

    # Iterate over the values and filter points outside the limits - DEBUGGING invalid values for Amplitude
    for x, y in zip(x_values, y_values):
        if x >= 0.05 and y >= 0.05:
            x_filtered.append(x)
            y_filtered.append(y)

    # Convert filtered lists to numpy arrays
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    # Perform linear regression only on filtered values
    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

    # Create a scatter plot with filtered values
    plt.scatter(x_filtered, y_filtered)

    # Set axis labels and title
    plt.xlabel('pyActigraphy ' + pyActigraphy_metric)
    plt.ylabel('CRAN ' + CRAN_metric)
    plt.title('kRA vs. L5')

    # Add a line of best fit (dotted) using filtered values
    line = slope * x_filtered + intercept
    plt.plot(x_filtered, line, 'r--')

    # Calculate R-squared only with filtered values
    r_squared = r_value ** 2

    # add a legend with R^2 and slope
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.4f}\nSlope: {slope:.4f}', ha='left', va='top',
             transform=plt.gcf().transFigure)

    # Determine the minimum and maximum values with a margin
    margin = 0.001  # Adjust the margin as needed
    min_value = min(min(x_values), min(y_values))
    max_value = max(max(x_values), max(y_values))
    xlim = (min_value * (1 - margin), max_value * (1 + margin))
    ylim = (min_value * (1 - margin), max_value * (1 + margin))

    # Set the limits for both axes with the margin
    #plt.xlim(xlim)
    #plt.ylim(ylim)

    # Save the figure to the specified filepath
    #plt.savefig(output_filepath)

    # Show the plot
    plt.show()


'''FUNCTION CALLS'''

# filepath, and individual filenames
filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/summary-metrics/'
pyActigraphy_filename = 'output_metrics_full_data_set.csv'
CRAN_filename = 'processedrhythms_n166_2023-06-19.csv'

# generate pyActigraphy dataframe
pyActigraphy_metrics_df = prepare_pyActigraphy_data(filepath, pyActigraphy_filename)

# generate CRAN dataframe
CRAN_metrics_df = prepare_CRAN_data(filepath, CRAN_filename)


# Generate the bias analysis figures
output_filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/summary-metrics/bias-figs/'

metric_to_compare = 'RA'
CRAN_vs_pyActigraphy_metric_bias(CRAN_metrics_df, pyActigraphy_metrics_df, metric_to_compare, output_filepath)

pyActigraphy_metric = 'kRA'
CRAN_metric = 'L5'
#CRAN_vs_pyActigraphy_two_metrics(CRAN_metrics_df, pyActigraphy_metrics_df, pyActigraphy_metric, CRAN_metric, output_filepath)

