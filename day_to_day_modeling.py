'''IMPORTS'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_data(subjective_data_filename, objective_data_filename, diagnosis_data_filename):
    """
    Read in csv files, and convert them to dataframes for data graphing.
    :param subjective_data_filename: the objective sleep data from REDCAP sleep questionnaires.
    :param objective_data_filename: the objective sleep data, from GGIR.
    :param diagnosis_data_filename: the file describing diagnosis and sex information.
    :return: subjective_sleep_df, objective_sleep_df, diagnosis_data : a tuple containing the three values respectively.
    """
    import_directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files'

    subjective_sleep_df = pd.read_csv(import_directory + subjective_data_filename)
    objective_sleep_df = pd.read_csv(import_directory + objective_data_filename)
    diagnosis_data_df = pd.read_csv(import_directory + diagnosis_data_filename)

    return subjective_sleep_df, objective_sleep_df, diagnosis_data_df


'''FUNCTION CALLS'''
subjective_data_filename = 'ActigraphyDatabase-FullSleepLogs_DATA_LABELS_2023-07-24_1554.csv'
objective_data_filename = 'part4_nightsummary_sleep_cleaned.csv'




