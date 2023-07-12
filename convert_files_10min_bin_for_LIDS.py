'''IMPORTS'''
import sys
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyActigraphy.analysis import LIDS   #LIDS tools import
import plotly.graph_objects as go

import os
import pandas as pd
import gzip
import shutil



import os
import pandas as pd

def convert_csv_files(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over the files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('-timeSeries.csv'):
            # Read the original CSV file
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)

            # Perform the data transformation
            window_size = 21  # Number of rows to sum up

            transformed_df = pd.DataFrame()
            transformed_df['time'] = df['time']
            transformed_df['acc'] = 0  # Initialize the 'acc' column

            for i in range(1, len(df), window_size):
                start_idx = i
                end_idx = min(i + window_size, len(df))
                window_sum = df['acc_med'][start_idx:end_idx].sum()
                transformed_df.loc[start_idx + 1, 'acc'] = window_sum

            # Save the transformed data to a new CSV file
            output_file_path = os.path.join(output_directory, f'{filename[:-4]}_converted_10min_LIDS.csv')
            transformed_df.to_csv(output_file_path, index=False)



def delete_files(input_directory):
    # Iterate over the files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('-timeSeries_converted_10min_LIDS.csv'):
            # Construct the file path
            file_path = os.path.join(input_directory, filename)

            # Delete the file
            os.remove(file_path)


def compress_csv(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as input_file, gzip.open(output_file_path, 'wb') as output_file:
        shutil.copyfileobj(input_file, output_file)

def compress_csvs_for_reading(input_directory):
    # Iterate over the files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('_converted_10min_LIDS.csv'):
            # Construct the file paths
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(input_directory, f'{filename}.gz')

            # Compress the CSV file, using the helper method
            compress_csv(input_file_path, output_file_path)





'''FUNCTION CALLS'''
input_directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'
output_directory = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

# Call the function to convert the CSV files
#convert_csv_files(input_directory, output_directory)
#delete_files(input_directory)
compress_csvs_for_reading(input_directory)
