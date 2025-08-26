'''IMPORTS'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os

def import_data(subjective_data_filename_all, subjective_data_filename_1, subjective_data_filename_2,
                objective_data_filename, diagnosis_data_filename, import_directory):
    """
    Read in csv files, and convert them to dataframes for data graphing.
    :param subjective_data_filename: the subjective sleep data from REDCAP sleep questionnaires. 1/2 -> RECAP files.
    :param objective_data_filename: the objective sleep data, from GGIR.
    :param diagnosis_data_filename: the file describing diagnosis and sex information.
    :return: subjective_sleep_df, objective_sleep_df, diagnosis_data : a tuple containing the three values respectively.
    """
    #import_directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
    #             'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

    # TEST COMMENT


    '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

    subjective_sleep_df_all = pd.read_csv(import_directory + subjective_data_filename_all)
    subjective_sleep_df_July_2023 = pd.read_csv(import_directory + subjective_data_filename_1)
    subjective_sleep_df_REDCAP_fixed = pd.read_csv(import_directory + subjective_data_filename_2)
    objective_sleep_df = pd.read_csv(import_directory + objective_data_filename)
    diagnosis_data_df = pd.read_csv(import_directory + diagnosis_data_filename)



    return (subjective_sleep_df_all, subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed,
    objective_sleep_df, diagnosis_data_df)

# add filename to each row in the dataframe where it is missing, with a dataframe with all rows from a single subject ID
def add_filename_per_subject_id(subject_id_df):
    # sort each group by the 'Repeat Instrument' column
    subject_id_df.sort_values('Repeat Instrument',  ascending=False, inplace=True)

    # filter rows that have a filename
    filename_rows = subject_id_df[subject_id_df['File Name'].notna()]
    n_filename_rows = len(filename_rows)
    # filter rows that have column 'Repeat Instrument' with value 'Daily Logs'
    daily_log_rows = subject_id_df[subject_id_df['Repeat Instrument'] == 'Daily Logs']
    if len(daily_log_rows) == 0:
        # if there are no daily log rows, return the dataframe unchanged
        return subject_id_df

    # separate the daily log rows into categories where values in 'Date' column are at least 1 month apart
    initial_date = daily_log_rows['Date'].iloc[0]
    # convert to datetime
    date_cursor = pd.to_datetime(initial_date, format='%m/%d/%y', errors='raise')
    # iterate over the rows in the daily log rows dataframe
    filename_row_index = 0
    for index, row in daily_log_rows.iterrows():
        # convert the date to datetime
        current_date = pd.to_datetime(row['Date'], format='%m/%d/%y', errors='raise')
        # check if the current date is after the date cursor
        if current_date < date_cursor:
            # if the current date is before the date cursor, skip this row (data is not in order), due to data entry error
            continue
        # check if the difference between the two dates is at least 30 days
        if (current_date - date_cursor).days >= 30:
            # get the next filename
            filename_row_index += 1
        subject_id_df.loc[index, 'File Name'] = filename_rows['File Name'].iloc[filename_row_index]
        date_cursor = current_date
        assert filename_row_index < n_filename_rows, "Filename index out of range. Check the number of filenames available."

    return  subject_id_df

def subjective_long_add_filename(subjective_sleep_df_all, filepath):
    """
    Add the filename to each row in the long format data table, so things can be matched with objective sleep data
    by filename and date.
    :param subjective_data_filename_all: the file containing subjective sleep metrics for all data.
    :return:
    """

    # add a row index column to the dataframe
    subjective_sleep_df_all['row_index'] = subjective_sleep_df_all.index
    # this is to help ensure that the rows are processed in the same order as in the original dataframe

    subject_id_dfs = dict(tuple(subjective_sleep_df_all.groupby(['Study ID'])))
    # create a dictionary of dataframes, one for each study ID
    # iterate over the dictionary and sort each dataframe by 'Repeat Instrument'
    for study_id, subject_id_df in subject_id_dfs.items():
        subject_id_df = add_filename_per_subject_id(subject_id_df)
        subject_id_dfs[study_id] = subject_id_df
    # convert the dictionary back to a single dataframe
    subjective_sleep_df_all = pd.concat(subject_id_dfs.values(), ignore_index=True)

    print(subjective_sleep_df_all)


    #1) keep iterating over rows, until a row is found that has a value for the filename
    for index, row in subjective_sleep_df_all.iterrows():
        filename = row['File Name']
        Study_ID = row['Study ID']
        if pd.notna(filename):
            # 2) add that filename to all the other rows in the df that have the same value for 'Study ID'
            for index_2, row_2 in subjective_sleep_df_all.iterrows():
                if pd.isna(row_2['File Name']) and row_2['Study ID'] == Study_ID:

                    # update filename
                    subjective_sleep_df_all.loc[index_2, 'File Name'] = filename

    # 3) delete the rows that have no values for excel columns L-Z (these only show the filename but contain no data).
    subset_columns = ['Overall quality', 'Deep Sleep', 'Well-rested', 'Mentally Alert']
    subjective_sleep_df_all = subjective_sleep_df_all.dropna(subset=subset_columns, how='all')

    # 4) export the modified df as a csv to the same directory, for viewing
    subjective_sleep_df_all.to_csv(filepath + 'ActigraphyDatabase-FullSleepLogs_ID_imputed.csv', index=False)

    # df is now ready to be merged with objective data
    return subjective_sleep_df_all

def reformat_date_european_to_american_objective(objective_sleep_df, filepath):
    """
    the objective sleep df has date formatted in European format. Reformat it to be in American format to be consistent.
    :param objective_sleep_df: the objective sleep data frame
    :return:
    """
    # Step 1: Use the str.replace() method to remove ".RData" from the "File Name" column. Reasign it to be in place.
    objective_sleep_df['filename'] = objective_sleep_df['filename'].str.replace(r'\.RData$', '', regex=True)

    # convert to date time
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

def reformat_date_to_common_format(dataframe, date_column):
    """
    HELPER: Reformat the date column to a common format in the dataframe.
    :param dataframe: The pandas DataFrame containing the date column.
    :param date_column: The name of the date column to be reformatted.
    :return: None
    """

    # Step 1: Convert the date_column to datetime with the format '%m/%d/%y' using coerce to handle invalid dates
    dataframe.loc[:, date_column] = pd.to_datetime(dataframe[date_column], format='%m/%d/%y', errors='coerce')

    # Step 2: Drop any NaT (Not a Time) values resulting from invalid date conversions
    dataframe.dropna(subset=[date_column], inplace=True)

    # Step 3: Convert the date_column to the desired format 'mm/dd/yy'
    dataframe.loc[:, date_column] = dataframe[date_column].dt.strftime('%m/%d/%y')

def merge_objective_subjective_files(subjective_sleep_df_all_fixed, objective_sleep_df_fixed):
    """
    Merge rows in the subjective and objective dfs by filename and date.
    :param subjective_sleep_df_all_fixed:
    :param objective_sleep_df_fixed:
    :return:
    """

    # reformat the date for subjective sleep df
    # date format from NEW data update june 25 2024: 2021-08-29
    # date format from PREVIOUS data: 8/29/21
    subjective_sleep_df_all_fixed['Date'] = pd.to_datetime(subjective_sleep_df_all_fixed['Date'], format='%m/%d/%y',
                                                         ) # errors = 'coerce'

    subjective_sleep_df_all_fixed['Date'] = subjective_sleep_df_all_fixed['Date'].dt.strftime('%m/%d/%y')

    # merge the dfs
    merged_df = pd.merge(subjective_sleep_df_all_fixed, objective_sleep_df_fixed, on=['File Name', 'Date'], how='inner')

    # add a sleep efficiency column
    merged_df['sleep_efficiency'] = merged_df['SleepDurationInSpt'] / merged_df['SptDuration']

    merged_df.to_csv(filepath + 'objective_subjective_merged_6-20-2025.csv', index=False)

    return merged_df

def merged_objsubj_agesexetiology(merged_df, age_sex_etiology_df):
    """
    merges the obj/subj df and the df containing info about age, sex, and etiology, by filename.
    :param merged_df:
    :param age_sex_etiology_df:
    :return:
    """

    # Perform the merge
    merged_df = pd.merge(merged_df, age_sex_etiology_df, left_on='File Name', right_on='Actigraphy_File', how='left')

    # write it to csv
    merged_df.to_csv(filepath + 'objective_subjective_merged_with_severity.csv', index=False)
    return merged_df



def plot_obj_vs_subjective_unbinned(merged_df, subj_x_value, obj_y_value, color, obj_y_axis_name):
    """
    Violin plot of objective sleep characteristic (x value) vs. subjective sleep characteristic (y value), unbinned.
    :param merged_df:
    :param x_value:
    :param y_value:
    :return:
    """

    # Create violin plots using Seaborn
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    sns.violinplot(x=merged_df[subj_x_value], y=merged_df[obj_y_value], color=color)


    # calculate medians
    medians = merged_df.groupby(subj_x_value)[obj_y_value].median()

    # Plot the white dots for the median values
    sns.stripplot(x=medians.index, y=medians.values, color='white', size=7, linewidth=2)

    # Add a line of best fit for the median values
    x_values = np.arange(1, len(medians)+1)  # Generate x values for the plot - 1 to 5 (centered with 1-5 subj. scores)
    coefficients = np.polyfit(x_values, medians.values, 1)  # Fit a first-degree polynomial (linear fit)

    # Calculate the y values for the line of best fit
    y_values = np.polyval(coefficients, x_values)

    # Find the x-axis position for the first and last violin plots
    x_first_violin = x_values[0] - 1.15
    x_last_violin = x_values[-1] - 0.75

    # Draw the line of best fit using plt.plot()
    plt.plot([x_first_violin, x_last_violin], [y_values[0], y_values[-1]], color='darkgrey', linestyle='dashed',
             linewidth=1.5)

    # Set the y-axis limits to include the range of the line
    plt.ylim(min(merged_df[obj_y_value] - 0.99), max(merged_df[obj_y_value] + 1.5))
    plt.xlim(-0.5, 4.6)

    # add slope, R^2 to the plot
    r_squared = np.corrcoef(x_values, medians.values)[0, 1] ** 2 # squaring the correlation of x_values and median values (R^2)
    plt.text(4.5, max(merged_df[obj_y_value] + 1) , f"R²: {r_squared:.2f}", fontsize=9, ha='right')

    plt.xlabel(subj_x_value)
    plt.ylabel(obj_y_axis_name)
    plt.show()


'''FUNCTION CALLS'''
subjective_data_filename_all = 'ActigraphyDatabase-FullSleepLogs_DATA_LABELS_2024-06-20_1036.csv'
subjective_data_filename_1 = 'ActigraphyDatabase-FullSleepLogs_DATA_LABELS_2024-06-20_1036.csv' # not in use
subjective_data_filename_2 = 'ActigraphyDatabase-FullSleepLogs_DATA_LABELS_2024-06-20_1036.csv' # not in use
objective_data_filename = 'part4_nightsummary_sleep_cleaned_June2024.csv'
diagnosis_data_filename = 'n196_combined_demographics_data_05-03-2025.csv'

# create virtual environment
# import_directory = os.environ["DAY_TO_DAY_MODELING_FILES"] # set virtual environment here
import_directory = './day-to-day-modeling-files-updated-data/'

# extract dataframes in a tuple
(subjective_sleep_df_all, subjective_sleep_df_July_2023, subjective_sleep_df_REDCAP_fixed,
    objective_sleep_df, diagnosis_data_df) = import_data(subjective_data_filename_all, subjective_data_filename_1, subjective_data_filename_2,
                objective_data_filename, diagnosis_data_filename, import_directory)

filepath = import_directory

##### CREATING THE MERGED DF WITH OBJECTIVE AND SUBJECTIVE DATA #####
subjective_sleep_df_all_fixed = subjective_long_add_filename(subjective_sleep_df_all, filepath)
objective_sleep_df_fixed = reformat_date_european_to_american_objective(objective_sleep_df, filepath)
merged_df = merge_objective_subjective_files(subjective_sleep_df_all_fixed, objective_sleep_df_fixed)
merged_df_final = merged_objsubj_agesexetiology(merged_df, diagnosis_data_df)

print('unique file names in subjective df: ')
subjective_data_df = pd.read_csv(import_directory + subjective_data_filename_all)
print(subjective_data_df['File Name'].nunique())

print('unique file names in objective df: ')
objective_data_df = pd.read_csv(import_directory + objective_data_filename)
print(objective_data_df['ID'].nunique())

print('unique fil names in final (merged) df: ')
print(merged_df_final['File Name'].nunique())

print("TESTING SOMETHING")

# Load data
subjective_data_df = pd.read_csv(import_directory + subjective_data_filename_all)
objective_data_df = pd.read_csv(import_directory + objective_data_filename)

# Remove '.cwa' extension from subjective file names
subjective_files = set(subjective_data_df['File Name'].str.replace('.cwa', '', regex=False).unique())
objective_files = set(objective_data_df['ID'].unique())

# Find file names in objective but not in subjective
objective_not_in_subjective = objective_files - subjective_files

# Print the result
print("File names in objective data but NOT in subjective data:")
for file_name in sorted(objective_not_in_subjective):
    print(file_name)

print(f"\nTotal: {len(objective_not_in_subjective)} files")

# Normalize the file names
subjective_files = set(subjective_data_df['File Name'].str.replace('.cwa', '', regex=False).unique())
objective_files = set(objective_data_df['ID'].unique())

# Find intersection (i.e., common file names)
files_in_both_obj_and_subj = subjective_files & objective_files

# Print count
print(f"Number of file names in BOTH objective and subjective data: {len(files_in_both_obj_and_subj)}")

# print the files that are present in both obj / subj file not not in the merged file
set_of_unique_merged_files = set(merged_df_final['File Name'].str.replace('.cwa', '', regex=False).unique())
print("files that are in both obj/subj but NOT in the merged final file: ")
print(files_in_both_obj_and_subj - set_of_unique_merged_files)













