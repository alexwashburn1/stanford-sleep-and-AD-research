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

'''FUNCTION CALLS'''
id = '79036_0000000504'
sleep_diary_filename = id+'.ods'
timeseries_filename = id+'-timeSeries.csv.gz'

# read in the raw data file
raw = read_input_data(timeseries_filename)

# try to read in the sleep diary
fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/'
sleep_diary = raw.read_sleep_diary(fpath + sleep_diary_filename)
print('name: ', raw.sleep_diary.name)
print('sleep diary: ', raw.sleep_diary.diary)

