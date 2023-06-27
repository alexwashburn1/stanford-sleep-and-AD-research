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
    Reads in a SINGLE csv.gz file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw


def crespo_AoT(raw):
    """
    Automatic identification of activity onset/offset times, based on the Crespo algorithm.
    :param raw: raw file to process
    :return: Arrays containing the estimated activity onset and offset times, respectively.
    """
    aot = raw.Crespo_AoT()
    return aot

def CK_0_1_classification(raw):
    """
    returns a time series with 0: sleep, 1: wake
    :param raw: the raw file to process
    :return: pandas core time series with 0 classified as sleep, 1 classified as wake
    """
    ck = raw.CK()
    return ck

def sadeh_0_1_classification(raw):
    """
    returns a time series with 0: sleep, 1: wake
    :param raw: the raw file to process
    :return: pandas core time series with 0 classified as sleep, 1 classified as wake
    """
    sadeh = raw.Sadeh()
    return sadeh

def crespo_0_1_classification(raw):
    """
    returns a time series with 0: sleep, 1: wake
    :param raw: the raw file to process
    :return: pandas core time series with 0 classified as sleep, 1 classified as wake
    """
    crespo = raw.Crespo()
    print('crespo AoT: ', raw.Crespo_AoT)
    return crespo

def sleep_profile(raw):
    """
    Normalized sleep daily profile
    :param raw: raw file to process
    :return:
    """
    sleep_prof = raw.SleepProfile()
    return sleep_prof

'''FUNCTION CALLS'''
filename = '78203_0000000613-timeSeries.csv.gz'
raw = read_input_data(filename)
aot = crespo_AoT(raw)
#print('aot: ', aot)
#print(aot[0])
#print('type of data: ', type(aot))

ck_classification_0_1 = CK_0_1_classification(raw)
#print('ck classification ', ck_classification_0_1)

ck_classification_0_1.to_csv('/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
               'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'
                             'ck_0_1_scoring_scoring_78203_0000000613.csv')

#sadeh_classification_0_1 = sadeh_0_1_classification(raw)
#print('sadeh classification: ', sadeh_classification_0_1)
#print('type sadeh: ', type(sadeh_classification_0_1))

#sadeh_classification_0_1.to_csv('/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
#                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/sadeh_0_1_scoring.csv')

#crespo_0_1_classification = crespo_0_1_classification(raw)
#print('crespo classification: ', crespo_0_1_classification)

#crespo_0_1_classification.to_csv('/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
#                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/crespo_0_1'
#                                 '_scoring_78203_0000000613.csv')

#sleep_prof = sleep_profile(raw)
#print('sleep profile: ', sleep_prof)



