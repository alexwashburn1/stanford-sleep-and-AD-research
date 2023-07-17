'''IMPORTS'''
import csv
import pandas as pd
import numpy as np


def convert_xlsx_to_csv_formatting(excel_file, REDCAP_csv_file, demographics_data_file):
    """
    A function that converts the format of the excel sleep log file to the format of the csv file (REDCAP).
    :param excel_file: excel file to convert format of
    :param csv_file: the csv file that has the appropriate format
    :param demographics_data_file: file that contains the filename corresponding to the subject IDs in the excel file.
    :return:
    """


    # read in the csv and xlsx files
    filepath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/subjective-sleep-logs/'
    excel_file_df = pd.read_excel(filepath + excel_file)
    REDCAP_csv_df = pd.read_csv(filepath + REDCAP_csv_file)
    demographics_data_df = pd.read_csv(filepath + demographics_data_file)

    # 0) fix formatting issues in the file

    # Find the row with a Subject ID value starting with '39000278'
    mask = excel_file_df['Subject ID'].astype(str).str.startswith('39000278')
    matched_row = excel_file_df[mask]
    print('matched_row: ', matched_row)

    excel_file_df.loc[mask, "Subject ID"] = 3900278.0

    # 1) Delete the column header "Subject ID" and instead add two new columns - 'ADRC ID' and 'SAMS PIDN'. Iterate over the
    # Subject ID column for the xlsx file and if the length  of the value is 3, add it to SAMS PIDN and add a NaN / blank value for the
    # ADRC ID. If the length is 7, add a NaN / blank value for the SAMS PIDN and add the value itself to the ADRC ID column.
    # Also, "fill in" the missing subject ID values.

    # Get the index of the 'Subject ID' column
    subject_id_index = excel_file_df.columns.get_loc('Subject ID')
    # Add 'ADRC ID' column to the right of 'Subject ID' column
    excel_file_df.insert(subject_id_index + 1, 'ADRC ID', np.nan)
    # Add 'SAMS PIDN' column to the right of 'ADRC ID' column
    excel_file_df.insert(subject_id_index + 2, 'SAMS PIDN', np.nan)

    # Iterate over 'Subject ID' column and update 'ADRC ID' and 'SAMS PIDN' based on conditions
    for index, value in excel_file_df['Subject ID'].items():
        if pd.notnull(value):  # If the value is not NaN
            if len(str(value)) >= 5 and len(str(value)) < 9:
                excel_file_df.at[index, 'SAMS PIDN'] = value
                previous_subject_id = value
                previous_column = 'SAMS PIDN'
            elif len(str(value)) == 9:
                excel_file_df.at[index, 'ADRC ID'] = value
                previous_subject_id = value
                previous_column = 'ADRC ID'
        else:  # If the value is NaN
            if previous_subject_id is not None:
                excel_file_df.at[index, previous_column] = previous_subject_id


    # 2) Rename the evening date column to 'Date'
    excel_file_df = excel_file_df.rename(columns={'Evening Date': 'Date'})

    # 3) Add the filename to the csv file to export
    # create a new column called 'filename', initialized with Nans at first
    SAMS_PIDN_index = excel_file_df.columns.get_loc('SAMS PIDN')
    excel_file_df.insert(SAMS_PIDN_index + 1, 'File Name', np.nan)

    # create a new column called 'Recruited Study', initialized with Nans at first
    excel_file_df.insert(0, 'Recruited Study', np.nan)

    # get the number of rows in the excel file df
    excel_file_num_rows = excel_file_df.shape[0]

    # iterate over the excel file df, filling in the filename values
    i = 1   # start at index one to avoid looking at the row with column names only.
    while i < excel_file_num_rows:
        if pd.isna(excel_file_df.at[i, 'ADRC ID']):
            # Must be a SAMS ID value
            SAMS_ID_value = excel_file_df.at[i, 'SAMS PIDN']

            # Find the row in demographics_data_df where SAMS_pidn matches SAMS ID value
            matched_row = demographics_data_df[demographics_data_df['SAMS_pidn'] == SAMS_ID_value]

            # Check if matched_row is not empty
            if not matched_row.empty:
                # Get the associated filename value from the first row in matched_row using iloc (and recruited study)
                filename_value = matched_row.iloc[0]['filename']
                recruited_study_value = matched_row.iloc[0]['Study']
                # Write the filename value to the 'filename' column in excel_file_df (and recruited study too)
                excel_file_df.loc[i, 'File Name'] = filename_value
                excel_file_df.loc[i, 'Recruited Study'] = recruited_study_value
            else:
                # Handle the case when no match is found
                filename_value = ''  # or any other desired handling
                recruited_study_value = ''

        else:
            # Must be an ADRC ID value
            ADRC_ID_value = excel_file_df.at[i, 'ADRC ID']


            # Find the row in demographics_data_df where adrc_id matches ADRC ID value
            matched_row = demographics_data_df[demographics_data_df['adrc_id'] == ADRC_ID_value]


            # Check if matched_row is not empty
            if not matched_row.empty:

                # Get the associated filename value from the first row in matched_row using iloc
                filename_value = matched_row.iloc[0]['filename']
                recruited_study_value = matched_row.iloc[0]['Study']
                # Write the filename value to the 'filename' column in excel_file_df
                excel_file_df.loc[i, 'File Name'] = filename_value
                excel_file_df.loc[i, 'Recruited Study'] = recruited_study_value
            else:

                # Handle the case when no match is found
                filename_value = ''  # or any other desired handling
                recruited_study_value = ''

        # Increment the index
        i += 1


    # 4) rename columns
    excel_file_df = excel_file_df.rename(columns={'1. Get into bed': 'Into Bed Time'})
    excel_file_df = excel_file_df.rename(columns={'3. Nap duration': 'Nap Time'})
    excel_file_df = excel_file_df.rename(columns={'4. Alcoholic drinks': 'Alcohol Consumption'})
    excel_file_df = excel_file_df.rename(columns={'5. Caffienated drinks': 'Caffeine Consumption'})
    excel_file_df = excel_file_df.rename(columns={'6. Sleep latency (mins)': 'Sleep Latency'})
    excel_file_df = excel_file_df.rename(columns={'7. # of Awakenings': 'Times Woken Up During Sleep'})
    excel_file_df = excel_file_df.rename(columns={'8.Difficulty falling asleep?': 'Had a hard time getting to sleep:'})
    excel_file_df = excel_file_df.rename(columns={'9. More awakenings than usual?': 'Woke up more frequently than usual during the night:'})
    excel_file_df = excel_file_df.rename(columns={'10. Difficulty falling back asleep?': 'If you woke up during the night, did you have difficulty getting back to sleep:'})
    excel_file_df = excel_file_df.rename(columns={'11. Earlier wake than normal?': 'Woke up earlier than you\'re usual time:'})
    excel_file_df = excel_file_df.rename(columns={'12. Earlier wake than normal and stayed awake?': 'Woke up earlier than usual and then couldn\'t get back to sleep:'})
    excel_file_df = excel_file_df.rename(columns={'13. Sleep quality': 'Overall quality'})
    excel_file_df = excel_file_df.rename(columns={'14. Depth of sleep': 'Deep Sleep'})
    excel_file_df = excel_file_df.rename(columns={'15. Well-rested?': 'Well-rested'})
    excel_file_df = excel_file_df.rename(columns={'16. Mentally alert?': 'Mentally Alert'})



    # add the missing filename value for the first row
    # Find the row with a Subject ID value starting with '39000278'
    mask_2 = excel_file_df['Sleep Latency'] == "10-15min"
    matched_row_2 = excel_file_df[mask_2]
    print('matched_row 2: ', matched_row_2)


    excel_file_df.loc[mask_2, 'File Name'] = '76576_0003900443.cwa'
    excel_file_df.loc[mask_2, 'Recruited Study'] = 'Tau PET'


    # 5) DROP UNNECCESARY COLUMNS
    # Assuming you have a DataFrame named df
    excel_file_df.drop("Subject ID", axis=1, inplace=True)
    excel_file_df.drop("2. Tried to go to sleep", axis=1, inplace=True)
    excel_file_df.drop("5. Wake up", axis=1, inplace=True)

    # 6) Add a column for recruited study


    ## EXPORT THE CSV ##
    excel_file_df.to_csv(filepath+'Sleep_Questionnaire_Data_Entry_July2023_REDCAP_fixed.csv', index=False)





'''FUNCTION CALLS'''

excel_file = 'Sleep Questionnaire Data Entry_July2023.xlsx'
REDCAP_csv_file = 'Full_Sleep_Logs_July2023.csv'
demographics_data_file = 'acti_n166.csv'

convert_xlsx_to_csv_formatting(excel_file, REDCAP_csv_file, demographics_data_file)

