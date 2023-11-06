#
# Convert Mendeley file from Winnebeck paper, into file format which can be read by pyActigraphy
import gzip
import shutil

import pandas as pd
import os


# make these paramters env variables TODO
# define the filepath
#fpath = './data/mendeley/'
fpath = os.environ.get('FILE_PATH')
filename = 'LIDS-sleep-bouts_database.csv'
output_path = fpath + 'converted/'
first_day_str = '2023-10-14 00:00:00' # make up a date, since the data doesn't have one
first_day = pd.to_datetime(first_day_str)
day = first_day
base_time = first_day
absolute_time = base_time
# create output path if it doesn't exist
import os
if not os.path.exists(output_path):
    os.makedirs(output_path)


# read in the file
mendeley_df = pd.read_csv(fpath + filename)

# list the columns
print(mendeley_df.columns)

# experiment filtering out certain groups
#mendeley_df.loc[mendeley_df["Group"] != "Children"]
#mendeley_df.loc[mendeley_df["Group"] != "Quilombola"]

# form dataframe with only the columns we want

# define the columns we want

columns_needed = ['ID', 'TimeSinceOnset_min', 'LIDS.raw']

mendeley_df = mendeley_df[columns_needed]

print(mendeley_df.head())

# split out dataframe by unique ID
unique_IDs = mendeley_df['ID'].unique()

# convert LIDS.raw column data back to activity levels, based on formula from Winnebeck paper
# from paper:  LIDS = 100/(activity count + 1)
# so
# activity count = 100/LIDS - 1

def convert_LIDS_to_activity(LIDS_raw):
    """
    Takes in a LIDS.raw value, and converts it to activity level
    :param LIDS_raw:
    :return: activity_count
    """
    if pd.isna(LIDS_raw) == True:
        return -1.0
    else:
        activity_count = 100.0/LIDS_raw - 1
        return activity_count

def convert_offset_minutes_to_absolute_time(offset_minutes):
    """
    Takes in a time offset in minutes, and converts it to absolute time
    :param offset_minutes:
    :return: absolute_time
    """

    global absolute_time, base_time
    #print("absolute_time start", absolute_time)

    # convert to seconds
    seconds = int(offset_minutes) * 60
    # add to start time
    if seconds == 0: # whenever we see offset == 0, add 1 day. to make more realistic time series for a person over several days
        base_time = absolute_time
    absolute_time = base_time + pd.to_timedelta(seconds, unit='s')
    #print("absolute_time end ", absolute_time)
    # convert to string in iso format
    return absolute_time

# convert data column in dataframe - apply conversion to every data value in LIDS raw column
mendeley_df['LIDS.raw'] = mendeley_df['LIDS.raw'].apply(convert_LIDS_to_activity)
# get rid of 0's
mendeley_df = mendeley_df[mendeley_df['LIDS.raw'] != -1.0]

# rename the column to 'acc'
mendeley_df.rename(columns={'LIDS.raw': 'acc'}, inplace=True)

# convert the time offset minutes column  to absolute time
#id = mendeley_df['TimeSinceOnset_min'].iloc[0] # get the first ID
#day = first_day # reset the day to the first day
mendeley_df['TimeSinceOnset_min'] = mendeley_df['TimeSinceOnset_min'].apply(convert_offset_minutes_to_absolute_time)
# rename   the TimeSinceOnset_min column to 'time'
mendeley_df.rename(columns={'TimeSinceOnset_min': 'time'}, inplace=True)

# add column called 'light' after column called 'acc' and set to 0.0
col_index= mendeley_df.columns.get_loc('acc')
col_index = col_index + 1
mendeley_df.insert(col_index, 'light', 0.0)
#
# # add column called 'moderate-vigorous' after column called 'light' and set to 0.0
col_index = col_index + 1
mendeley_df.insert(col_index, 'moderate-vigorous', 0.0)
#
# # add column called 'sedentary' after column called 'moderate-vigorous' and set to 0.0
col_index = col_index + 1
mendeley_df.insert(col_index, 'sedentary', 0.0)
#
# # add column called 'sleep' after column called sedentary'' and set to 0.0
col_index = col_index + 1
mendeley_df.insert(col_index, 'sleep', 0.0)
#
# # add column called 'MET' after column called sleep'' and set to 0.0
col_index = col_index + 1
mendeley_df.insert(col_index, 'MET', 0.0)


# Iterate to export a unique CSV file for each user ID
for ID in unique_IDs:
    #get the data for this ID
    ID_df = mendeley_df[mendeley_df['ID'] == ID]

    print(ID_df.head())
    print(ID_df.shape)
    # remove the ID column: to get appropriate raw data format (excluding ID column)
    ID_df = ID_df.drop(columns=['ID'])
    # save the data to a csv file
    ID_df.to_csv(output_path + str(ID) + '.csv', index=False)

    # compress files
    with open(output_path + str(ID) + '.csv', 'rb') as f_in:
        with gzip.open(output_path + str(ID) + '.csv.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)





