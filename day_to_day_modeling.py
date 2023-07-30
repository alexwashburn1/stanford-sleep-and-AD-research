'''IMPORTS'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

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
    subjective_sleep_df_all_fixed['Date'] = pd.to_datetime(subjective_sleep_df_all_fixed['Date'], format='%m/%d/%y',
                                                         errors='coerce')

    subjective_sleep_df_all_fixed['Date'] = subjective_sleep_df_all_fixed['Date'].dt.strftime('%m/%d/%y')


    # merge the dfs
    merged_df = pd.merge(subjective_sleep_df_all_fixed, objective_sleep_df_fixed, on=['File Name', 'Date'], how='inner')

    # add a sleep efficiency column
    print('adding column ')
    merged_df['sleep_efficiency'] = merged_df['SleepDurationInSpt'] / merged_df['SptDuration']
    print('added column')

    merged_df.to_csv(filepath + 'objective_subjective_merged.csv', index=False)

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

    #### RELEVANT THRESHOLDS ####
    # 1) Sleep Efficiency:
    #  x_first_violin = x_values[0] - 1.15
    #  x_last_violin = x_values[-1] - 0.75
    #  plt.ylim(min(merged_df[obj_y_value] - 0.13), max(merged_df[obj_y_value] + 0.15))
    #  plt.xlim(-0.5, 4.6)
    #  plt.text(4.5, max(y_values) + 0.2, f"R²: {r_squared:.2f}", fontsize=9, ha='right')




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

##### CREATING THE MERGED DF WITH OBJECTIVE AND SUBJECTIVE DATA #####
#subjective_sleep_df_all_fixed = subjective_long_add_filename(subjective_sleep_df_all, filepath)
#objective_sleep_df_fixed = reformat_date_european_to_american_objective(objective_sleep_df, filepath)
#merged_df = merge_objective_subjective_files(subjective_sleep_df_all_fixed, objective_sleep_df_fixed)

##### OBJECTIVE VS SUBJECTIVE PLOTS #####
# read in the merged data frame
merged_df = pd.read_csv(filepath + 'objective_subjective_merged.csv')

# plot
subj_characteristics = ['Deep Sleep', 'Overall quality', 'Well-rested', 'Mentally Alert']
color = 'darkgoldenrod'
obj_y_value = 'WASO'
obj_y_axis_name = 'WASO'
for subj_x_value in subj_characteristics:
    plot_obj_vs_subjective_unbinned(merged_df, subj_x_value, obj_y_value, color, obj_y_axis_name)

plt.show()








