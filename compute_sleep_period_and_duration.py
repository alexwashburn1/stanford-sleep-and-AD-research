''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo    # debug import
import os
import re
import pyexcel_ods3
from datetime import timedelta
from dateutil import parser



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

def ck_0_1_classification_single_file(raw, output_path, file_ID, threshold=0.1): # TODO - can change the threshold
    """
    Generates the binarized classification for sleep and wake (Cole-Kripke) for a particular raw file.
    :param raw: the raw file
    """

    ck = raw.CK(threshold = threshold)
    # write the binarized csv to the correct output path
    thresh_str = str(threshold)
    ck.to_csv(output_path + 'ck_0_1_scoring_scoring_' + file_ID + '_' + thresh_str + '.csv')

def sadeh_0_1_classification_single_file(raw, output_path, file_ID):
    """
    Generates the binarized classification for sleep and wake (Cole-Kripke) for a particular raw file.
    :param raw: the raw file
    """

    sadeh = raw.Sadeh()
    # write the binarized csv to the correct output path
    sadeh.to_csv(output_path + 'sadeh_0_1_scoring_scoring_' + file_ID + '.csv')


def sleep_0_1_classification_all_files(raw_input_path, sleep_diary_path, output_path, sadeh=False, ck=True):
   """
   Determines all of the raw files to classify by looking for which ones have a valid sleep log.
    After determining which ones have a valid sleep log, creates a raw object for them, and based on the raw object
    creates a binarized ck object which logs each epoch as sleep or non-sleep. Writes that to a new folder.
   :param raw_input_path: the input path of the raw files (.csv.gz)
   :param sleep_diary_path: the directory of the sleep diaries
   :param output_path: the path to write the binarized ck objects to
   :param sadeh: if true, 0 1 classification for sadeh will be generated. if false, then not.
   :param ck: if true, 0 1 classification for ck will be generated. if false, then not.
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
       # look at sadeh and ck parameters to see which one will be created.
       if ck == True and sadeh == False:
           ck_0_1_classification_single_file(raw, output_path, file_ID)
       else:
           if sadeh == True and ck == False:
               sadeh_0_1_classification_single_file(raw, output_path, file_ID)



from datetime import timedelta
import re
import pandas as pd
import pyexcel_ods3

def calculate_sleep_period(ck_0_1_file, sleep_journal_file, ck_file_path, sleep_journal_file_path):
    """
    Iterates over a single 0 1 ck sleep classification file, and determines the sleep period based on the start and
    stop times dictated by the sleep journal file.
    :return:
    """

    # read in the ck_0_1 file, and convert it to a dataframe
    ck_df = pd.read_csv(ck_file_path + ck_0_1_file)

    # Remove the trailing part from the "time" column using regular expressions
    ck_df['time'] = ck_df['time'].apply(lambda x: re.sub(r'\:\d{2}-\d{2}:\d{2}$', '', x))
    print('ck df: ', ck_df)

    # read in the sleep journal file, and convert it to a data frame
    data = pyexcel_ods3.get_data(sleep_journal_file_path + sleep_journal_file)
    sleep_journal_df = pd.DataFrame(data[next(iter(data))])
    print('sleep_journal_df: ', sleep_journal_df)

    # Create a new DataFrame to store the sleep period details
    result_df = pd.DataFrame(columns=['night', 'computed duration', 'journal logged sleep duration'])

    # Iterate over each row in 'sleep_journal_df' starting from the 4th row
    for i, journal_row in enumerate(sleep_journal_df.iloc[3:].itertuples(index=False), start=1):
        journal_start = journal_row[1]  # Access the second column (index 1)
        journal_end = journal_row[2]  # Access the third column (index 2)

        # Find the row in 'ck_df' where 'time' matches the 'START' value of the sleep journal
        start_row = ck_df[ck_df['time'] == journal_start].iloc[0]

        # Find the row in 'ck_df' where 'time' matches the 'END' value of the sleep journal
        end_row = ck_df[ck_df['time'] == journal_end].iloc[0]

        # Calculate the duration by counting the number of rows with 'acc' value equal to 1
        duration = ck_df.loc[start_row.name:end_row.name, 'acc'].eq(1).sum() - 1

        # Convert duration to timedelta format
        duration_timedelta = timedelta(minutes=int(duration))

        # Format the duration as hh:mm
        duration_formatted = str(duration_timedelta)

        # Calculate the journal logged sleep duration
        journal_duration = str(parser.parse(journal_end) - parser.parse(journal_start))

        # Add the sleep period details to the result DataFrame
        result_df = result_df.append({'night': i, 'computed duration': duration_formatted, 'journal logged sleep duration': journal_duration}, ignore_index=True)

    # Get the ID from the sleep_journal_file
    id = sleep_journal_file.split('.')[0]

    # Construct the output file path
    sadeh_or_ck = 'ck' # NOTE - can change to whichever is being generated. TODO - add this as a parameter.
    output_file_path = os.path.join(ck_file_path, f"{id}_sleep_duration_metrics_" + sadeh_or_ck + "_0.1.csv")

    # Write the result DataFrame to CSV
    result_df.to_csv(output_file_path, index=False)

    # Print the result DataFrame
    print(result_df)




'''FUNCTION CALLS'''

raw_input_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

sleep_diary_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/'

output_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/ck_sleep_0_1_scoring/'


#sleep_0_1_classification_all_files(raw_input_path, sleep_diary_path, output_path)

file_id = '78203_0000000534'

sadeh_or_ck = 'ck'

calculate_sleep_period(sadeh_or_ck + '_0_1_scoring_scoring_' + file_id + '_0.1.csv', file_id + '.ods', output_path, sleep_diary_path)











# OLD:
# # DETERMINE SLEEP PERIOD
    #
    # # Iterate over each row in 'ck_df'
    # for ck_row in ck_df.itertuples(index=True):
    #     ck_time = ck_row.time
    #
    #     # Flag variable to control the iteration over 'sleep_journal_df'
    #     match_found = False
    #
    #     # Iterate over each row in 'sleep_journal_df' only if a match is not found
    #     for journal_row in sleep_journal_df.iloc[3:].itertuples(index=False):
    #         journal_start = journal_row[1]  # Access the second column (index 1)
    #
    #         # Check if the value for 'time' column in 'ck_df' matches the 'START' value in 'sleep_journal_df'
    #         if ck_time == journal_start:
    #             match_found = True
    #             print(True)
    #             print(ck_time)
    #             print(journal_start)
    #
    #             # Get the index of the matched row in 'ck_df'
    #             ck_row_index = ck_row.Index
    #
    #             # Check if there are at least 15 rows before the matched row
    #             if ck_row_index >= 15:
    #                 # Flag variable to track three consecutive rows with 'acc' value equal to 1
    #                 consecutive_found = False
    #
    #
    #                 # Iterate forward from 15 rows before the matched row
    #                 for i in range(ck_row_index - 15, ck_row_index + 1):
    #                     if ck_df.loc[i, 'acc'] == 1:
    #                         if not consecutive_found:
    #                             # Start of a potential three consecutive rows with 'acc' value equal to 1
    #                             consecutive_found = True
    #                             start_index = i
    #                     elif consecutive_found:
    #                         # Check if three consecutive rows have 'acc' value equal to 1
    #                         if i - start_index >= 2:
    #                             start_time = ck_df.loc[start_index, 'time']
    #                             #print("Start Time:", start_time)
    #
    #                             # Print the first row of the three consecutive rows
    #                             print('first row of the three consecutive rows: ', ck_df.iloc[start_index])
    #
    #                             break
    #                         else:
    #                             # Reset flag if consecutive rows are not enough
    #                             consecutive_found = False
    #
    #                 # Check if three consecutive rows were not found
    #                 if not consecutive_found:
    #                     print("No three consecutive rows found with 'acc' value equal to 1.")
    #
    #             else:
    #                 print("Not enough rows before the matched row.")
    #
    #             break
    #
    #     # If a match is not found, print False and proceed to the next row in 'ck_df'
    #     if not match_found:
    #         continue






