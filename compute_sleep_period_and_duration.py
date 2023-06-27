''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo    # debug import
import os
import re

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

def ck_0_1_classification_single_file(raw, output_path, file_ID):
    """
    Generates the binarized classification for sleep and wake (Cole-Kripke) for a particular raw file.
    :param raw: the raw file
    """

    ck = raw.CK()
    # write the binarized csv to the correct output path
    ck.to_csv(output_path + 'ck_0_1_scoring_scoring_' + file_ID + '.csv')


def ck_0_1_classification_all_files(raw_input_path, sleep_diary_path, output_path):
   """
   Determines all of the raw files to classify by looking for which ones have a valid sleep log.
    After determining which ones have a valid sleep log, creates a raw object for them, and based on the raw object
    creates a binarized ck object which logs each epoch as sleep or non-sleep. Writes that to a new folder.
   :param raw_input_path: the input path of the raw files (.csv.gz)
   :param sleep_diary_path: the directory of the sleep diaries
   :param output_path: the path to write the binarized ck objects to
   """

   # determine which IDs have a valid sleep journal
   file_extension = '.ods'

   # add them to a list
   file_names = os.listdir(sleep_diary_path)
   file_names = [file_name for file_name in file_names if file_name.endswith(file_extension)]

   # Extract characters before '.ods' using regular expressions
   pattern = r'^(.*?)' + re.escape(file_extension)
   file_names_without_extension = [re.match(pattern, file_name).group(1) for file_name in file_names if not file_name.startswith('~')]

   # for each ID in file names, create a raw object, create a ck object of the raw object, and write it to the output csv.
   for file_ID in file_names_without_extension:
       full_file_name = file_ID + '-timeSeries.csv.gz'
       raw = read_input_data(full_file_name)
       # create 0 1 binarized sleep scoring for the file, and write it to the output csv path
       ck_0_1_classification_single_file(raw, output_path, file_ID)


def calculate_sleep_period()


'''FUNCTION CALLS'''

raw_input_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

sleep_diary_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/'

output_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/ck_sleep_0_1_scoring/'

ck_0_1_classification_all_files(raw_input_path, sleep_diary_path, output_path)


