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


def read_and_prepare_GGIR_data(filepath, GGIR_filename):
    """
    Prepares the GGIR csv data to be compared to the pyActigraphy data, namely by altering column names and sorting.
    :param filepath: the filepath to the GGIR csv file.
    :param GGIR_filename:
    :return: GGIR_metrics_df: a dataframe with the altered GGIR data.
    """

    # read in the GGIR summary metrics data
    GGIR_metrics_df = pd.read_csv(filepath + GGIR_filename)

    # rename columns: 'IV_intradailyvariability' to 'IV', and 'IS_interdailystability' to 'IS'
    GGIR_metrics_df.rename(columns={'IV_intradailyvariability': 'IV', 'IS_interdailystability': 'IS'}, inplace=True)

    # Sort the "filename" column in pyActigraphy_metrics_df in ascending order.
    GGIR_metrics_df = GGIR_metrics_df.sort_values("filename", ascending=True)
    print('GGIR: ', GGIR_metrics_df.head())

    return GGIR_metrics_df

def read_and_prepare_CRAN_data(filepath, CRAN_filename):
    """
     Prepares the CRAN csv data to be compared to the CRAN data, namely by altering column names and sorting.
    :param filepath: the filepath to the CRAN csv file.
    :param CRAN_filename: the filename of the CRAN file (.csv)
    :return: CRAN_metrics_df: a dataframe with the altered CRAN data.
    """
    # read in summary metrics from CRAN
    CRAN_metrics_df = pd.read_csv(filepath + CRAN_filename)

    # Sort the filename column in CRAN_metrics_df in ascending order
    CRAN_metrics_df = CRAN_metrics_df.sort_values("filename", ascending=True)
    print('CRAN: ', CRAN_metrics_df.head())

    return CRAN_metrics_df


def read_and_prepare_pyActigraphy_data(filepath, pyActigraphy_filename):
    """
    Prepares the pyActigraphy csv data to be compared to the CRAN data, namely by altering column names and sorting.
    :param filepath: the filepath to the pyActigraphy csv file.
    :param pyActigraphy_filename: the filename of the pyActigraphy file (.csv)
    :return: pyActigraphy_metrics_df: a dataframe with the altered pyActigraphy data.
    """
    # read in summary metrics from pyActigraphy
    pyActigraphy_metrics_df = pd.read_csv(filepath + pyActigraphy_filename)

    # Sort the "filename" column in pyActigraphy_metrics_df in ascending order.
    pyActigraphy_metrics_df = pyActigraphy_metrics_df.sort_values("filename", ascending=True)
    print('pyActigraphy: ', pyActigraphy_metrics_df.head())

    return pyActigraphy_metrics_df



def IS_vs_IV(package_name, package_df):
    """
    Plot of intradaily variability vs. interdaily stability, for a given package
    :param package_name: a string describing the name of the package used
    :param package_df: a dataframe containing data for the package used.
    :return:
    """

    # extract IV, IS values
    IV_values = package_df['IV']    # x
    IS_values = package_df['IS']    # y

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(IV_values, IS_values)

    # Create a scatter plot with filtered values
    plt.scatter(IV_values, IS_values)

    # Set axis labels and title
    plt.xlabel('IV')
    plt.ylabel('IS')
    plt.title('Comparison of IS vs IV for {}'.format(package_name))

    # Add a line of best fit (dotted) using filtered values
    line = slope * IV_values + intercept
    plt.plot(IV_values, line, 'r--')

    # Calculate R-squared only with filtered values
    r_squared = r_value ** 2

    # Add R-squared value and slope to the plot
    # Add R-squared value and slope to the plot
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.4f}\nSlope: {slope:.4f}', ha='left', va='top',
             transform=plt.gcf().transFigure)

    # Determine the minimum and maximum values with a margin
    margin = 0.5  # Adjust the margin as needed
    min_value = min(min(IV_values), min(IS_values))
    max_value = max(max(IV_values), max(IS_values))
    xlim = (min_value * (1 - margin), max_value * (1 + margin))
    ylim = (min_value * (1 - margin), max_value * (1 + margin))

    # Set the limits for both axes with the margin
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Show the plot
    plt.show()


def kRA_vs_IV(package_name, package_df):
    """
    ONLY WORKS FOR PYACTIGRAPHY PACKAGE. Plot of kRA vs. IV
    :param package_name: the name of the package - must be "pyActigraphy"
    :param package_df: the dataframe containing the pyActigraphy data.
    :return:
    """

    # extract kRA, IV values
    IV_values = package_df['IV']  # x
    kRA_values = package_df['kRA']  # y

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(IV_values, kRA_values)

    # Create a scatter plot with filtered values
    plt.scatter(IV_values, kRA_values)

    # Set axis labels and title
    plt.xlabel('IV')
    plt.ylabel('kRA')
    plt.title('Comparison of IV vs kRA for {}'.format(package_name))

    # Add a line of best fit (dotted) using filtered values
    line = slope * IV_values + intercept
    plt.plot(IV_values, line, 'r--')

    # Calculate R-squared only with filtered values
    r_squared = r_value ** 2

    # Add R-squared value and slope to the plot
    # Add R-squared value and slope to the plot
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.4f}\nSlope: {slope:.4f}', ha='left', va='top',
             transform=plt.gcf().transFigure)

    # Determine the minimum and maximum values with a margin
    margin = 0.5  # Adjust the margin as needed
    min_value = min(min(IV_values), min(kRA_values))
    max_value = max(max(IV_values), max(kRA_values))
    xlim = (min_value * (1 - margin), max_value * (1 + margin))
    ylim = (min_value * (1 - margin), max_value * (1 + margin))

    # Set the limits for both axes with the margin
    #plt.xlim(xlim)
    #plt.ylim(ylim)

    # Show the plot
    plt.show()

def kRA_vs_RA(package_name, package_df):
    """
    ONLY WORKS FOR PYACTIGRAPHY PACKAGE. Plot of kRA vs. IV
    :param package_name: the name of the package - must be "pyActigraphy"
    :param package_df: the dataframe containing the pyActigraphy data.
    :return:
    """

    # extract kRA, IV values
    RA_values = package_df['RA']  # x
    kRA_values = package_df['kRA']  # y

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(RA_values, kRA_values)

    # Create a scatter plot with filtered values
    plt.scatter(RA_values, kRA_values)

    # Set axis labels and title
    plt.xlabel('Relative Amplitude (RA)')
    plt.ylabel('kRA')
    plt.title('Comparison of Relative Amplitude vs kRA for {}'.format(package_name))

    # Add a line of best fit (dotted) using filtered values
    line = slope * RA_values + intercept
    plt.plot(RA_values, line, 'r--')

    # Calculate R-squared only with filtered values
    r_squared = r_value ** 2

    # Add R-squared value and slope to the plot
    # Add R-squared value and slope to the plot
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.4f}\nSlope: {slope:.4f}', ha='left', va='top',
             transform=plt.gcf().transFigure)

    # Determine the minimum and maximum values with a margin
    margin = 0.5  # Adjust the margin as needed
    min_value = min(min(RA_values), min(kRA_values))
    max_value = max(max(RA_values), max(kRA_values))
    xlim = (min_value * (1 - margin), max_value * (1 + margin))
    ylim = (min_value * (1 - margin), max_value * (1 + margin))

    # Set the limits for both axes with the margin
    #plt.xlim(xlim)
    #plt.ylim(ylim)

    # Show the plot
    plt.show()


'''FUNCTION CALLS'''

# relevant files
# filepath, and individual filenames
filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/summary-metrics/'
GGIR_filename = 'GGIR_summary_n166.csv'
CRAN_filename = 'processedrhythms_n166_2023-06-19.csv'
pyActigraphy_filename = 'output_metrics_full_data_set.csv'


pyActigraphy_df = read_and_prepare_pyActigraphy_data(filepath, pyActigraphy_filename)
GGIR_df = read_and_prepare_GGIR_data(filepath, GGIR_filename)
CRAN_df = read_and_prepare_CRAN_data(filepath, CRAN_filename)

package_name = 'GGIR'
package_df = GGIR_df

IS_vs_IV(package_name, package_df)
#kRA_vs_IV(package_name, package_df)
#kRA_vs_RA(package_name, package_df)

