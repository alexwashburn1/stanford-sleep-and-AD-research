'''IMPORTS'''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyActigraphy.analysis import LIDS   #LIDS tools import

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

    # transform LIDS data using a LIDS transformation
    LIDS.lids_transform(LIDS_obj, ts=raw)



''' FUNCTION CALLS '''

filename = '67067_0000000131-timeSeries.csv.gz'
raw = read_input_data(filename)

# Create a LIDS object
test_lids_obj = LIDS()

# test run for debug
LIDS_functionality_test(test_lids_obj, raw)










