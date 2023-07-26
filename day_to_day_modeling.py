'''IMPORTS'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def import_data(subjective_data_filename_all, subjective_data_filename_1, subjective_data_filename_2,
                objective_data_filename, diagnosis_data_filename):
    """
    Read in csv files, and convert them to dataframes for data graphing.
    :param subjective_data_filename: the objective sleep data from REDCAP sleep questionnaires. 1/2 -> RECAP files.
    :param objective_data_filename: the objective sleep data, from GGIR.
    :param diagnosis_data_filename: the file describing diagnosis and sex information.
    :return: subjective_sleep_df, objective_sleep_df, diagnosis_data : a tuple containing the three values respectively.
    """
    import_directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

    subjective_sleep_df_all = pd.read_csv(import_directory + subjective_data_filename_all)
    subjective_sleep_df_July_2023 = pd.read_csv(import_directory + subjective_data_filename_1)
    subjective_sleep_df_REDCAP_fixed = pd.read_csv(import_directory + subjective_data_filename_2)
    objective_sleep_df = pd.read_csv(import_directory + objective_data_filename)
    diagnosis_data_df = pd.read_csv(import_directory + diagnosis_data_filename)



    return (subjective_sleep_df_all, subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed,
    objective_sleep_df, diagnosis_data_df)

def subjective_long_add_filename(subjective_sleep_df_all, filepath):
    """
    Add the filename to each row in the long format data table, so things can be matched with objective sleep data
    by filename and date.
    :param subjective_data_filename_all: the file containing subjective sleep metrics for all data.
    :return:
    """

    #1) keep iterating over rows, until a row is found that has a value for the filename
    for index, row in subjective_sleep_df_all.iterrows():
        print('file name: ', row['File Name'])
        filename = row['File Name']
        Study_ID = row['Study ID']
        if pd.notna(filename):
            print('filename: ', filename)
            # 2) add that filename to all the other rows in the df that have the same value for 'Study ID'
            for index_2, row_2 in subjective_sleep_df_all.iterrows():

                if pd.isna(row_2['File Name']) and row_2['Study ID'] == Study_ID:
                    print('entered!')

                    # update filename
                    subjective_sleep_df_all.loc[index_2, 'File Name'] = filename

    # 3) delete the rows that have no values for excel columns L-Z (these only show the filename but contain no data).
    subset_columns = ['Overall quality', 'Deep Sleep', 'Well-rested', 'Mentally Alert']
    subjective_sleep_df_all = subjective_sleep_df_all.dropna(subset=subset_columns, how='all')

    # 4) export the modified df as a csv to the same directory, for viewing
    print('exporting')
    subjective_sleep_df_all.to_csv(filepath + 'ActigraphyDatabase-FullSleepLogs_ID_imputed.csv', index=False)

    # df is now ready to be merged with objective data
    return subjective_sleep_df_all

def reformat_date_european_to_american_objective(objective_sleep_df, filepath):
    """
    the objective sleep df has date formatted in European format. Reformat it to be in American format to be consistent.
    :param objective_sleep_df: the objective sleep data frame
    :return:
    """
    # Step 1: Use the str.replace() method to remove ".RData" from the "File Name" column
    objective_sleep_df['filename'].str.replace(r'\.RData$', '', regex=True)

    objective_sleep_df['calendar_date'] = pd.to_datetime(objective_sleep_df['calendar_date'], format='%d/%m/%Y',
                                                         errors='coerce')

    # Step 2: Convert the "calendar_date" column to the desired format 'mm/dd/yy'
    objective_sleep_df['calendar_date'] = objective_sleep_df['calendar_date'].dt.strftime('%m/%d/%y')

    for index_a, row_a in objective_sleep_df.iterrows():
        print('date: ', row_a['calendar_date'])

    #rename the 'calendar_date' column to 'Date'
    objective_sleep_df.rename(columns={"calendar_date": "Date", "filename": "File Name"}, inplace=True)



    # export
    objective_sleep_df.to_csv(filepath + 'part4_nightsummary_sleep_cleaned_fixed_date.csv', index=False)
    return objective_sleep_df

def merge_objective_subjective_files(subjective_sleep_df_all_fixed, objective_sleep_df_fixed):
    """
    Merge rows in the subjective and objective dfs by filename and date.
    :param subjective_sleep_df_all_fixed:
    :param objective_sleep_df_fixed:
    :return:
    """
    merged_df = pd.merge(subjective_sleep_df_all_fixed, objective_sleep_df_fixed, on=['File Name', 'Date'], how='inner')
    merged_df.to_csv(filepath + 'objective_subjective_merged.csv', index=False)


'''FUNCTION CALLS'''
subjective_data_filename_all = 'ActigraphyDatabase-FullSleepLogs_DATA_LABELS_2023-07-24_1554.csv'
subjective_data_filename_1 = 'Full_Sleep_Logs_July2023.csv'
subjective_data_filename_2 = 'Sleep_Questionnaire_Data_Entry_July2023_REDCAP_fixed.csv'
objective_data_filename = 'part4_nightsummary_sleep_cleaned.csv'
diagnosis_data_filename = 'AgeSexDx_n166_2023-07-13.csv'

# extract dataframes in a tuple
(subjective_sleep_df_all, subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed,
    objective_sleep_df, diagnosis_data_df) = import_data(subjective_data_filename_all, subjective_data_filename_1, subjective_data_filename_2,
                objective_data_filename, diagnosis_data_filename)

filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

subjective_sleep_df_all_fixed = subjective_long_add_filename(subjective_sleep_df_all, filepath)
objective_sleep_df_fixed = reformat_date_european_to_american_objective(objective_sleep_df, filepath)
merge_objective_subjective_files(subjective_sleep_df_all_fixed, objective_sleep_df_fixed)







