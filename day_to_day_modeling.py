'''IMPORTS'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def merge_two_REDCAP_files(subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed):
    """
    adds any of the lines from the REDCAP fixed file to the end of the July 2023 file, under the condition that
    there is not a corresponding row with the filename and date already.
    :param subjective_sleep_df_July_2023: the REDCAP file that was already in proper format
    :param subjective_sleep_df_REDCAP_fixed: the REDCAP file that I personally fixed
    :return: merged df - July 2023 df that has extra rows at the end of it from REDCAP file that are new.
    """

    # Step 1: Define the possible date formats
    date_formats = ['%m/%d/%y', '%m/%d/%y %H:%M']

    # Step 2: Convert 'date' column in subjective_sleep_df_July_2023 to datetime
    subjective_sleep_df_July_2023['Date'] = pd.to_datetime(subjective_sleep_df_July_2023['Date'], errors='coerce')

    # Step 3: Convert 'date' column in subjective_sleep_df_REDCAP_fixed using multiple formats
    for date_format in date_formats:
        converted_dates = pd.to_datetime(subjective_sleep_df_REDCAP_fixed['Date'], format=date_format, errors='coerce')
        if not converted_dates.isna().all():
            # At least one format succeeded, update the 'date' column
            subjective_sleep_df_REDCAP_fixed['Date'] = converted_dates
            break

    # 2) iterate over rows in the REDCAP fixed file, and check if there is a match for both date and filename values
    print('length of the July 2023 file: ', len(subjective_sleep_df_July_2023))
    all_matching_rows = []
    for index, row in subjective_sleep_df_REDCAP_fixed.iterrows():
        # extract row and date from REDCAP fixed file
        filename = row['File Name']
        date = row['Date']

        # check for matches in July 2023 file
        matching_rows = subjective_sleep_df_July_2023[
            (subjective_sleep_df_July_2023['File Name'] == filename) &
            (subjective_sleep_df_July_2023['Date'] == date)
            ]
        all_matching_rows.append(matching_rows)

        if matching_rows.empty:
            print(f'no matching rows found for {filename} on {date}.')

        elif not matching_rows.empty:
            print(f'match found for {filename} on {date}:')
            print(matching_rows)




    # 2) if there is a line in the July 2023 file that has a matching date and matching file ID, skip the REDCAP row
    # 3) if not, append the REDCAP row to the end of the file




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


merge_two_REDCAP_files(subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed)




