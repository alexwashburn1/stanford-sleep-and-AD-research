import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os


def rename_acc_MED(input_dir, output_dir, filename):
    """
    Helper function to delete the acc column that does not have imputed values. Take the acc_med column which has imputed values
    calculated already. Rename the acc_med column to acc, so pyActigraphy recognizes it.
    :param csv: the csv file to modify
    :return:
    """

    # import the csv file, convert to dataframe
    df = pd.read_csv(input_dir+filename)

    # drop the acc (non-impute) column
    df.drop(columns=['acc'], inplace=True)

    # rename the acc_MED to acc
    df.rename(columns={'acc_med': 'acc'}, inplace=True)

    # write out the new csv file to the folder, as a compressed csv file
    df.to_csv(output_dir+filename+'.gz', compression='gzip', index=False)


def rename_acc_batch_csv(input_dir, output_dir):
    """
    iterates over all files in a directory, renames acc_med to acc and deletes OG acc
    :return:
    """

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            rename_acc_MED(input_dir, output_dir, filename)
            continue
        else:
            continue


'''Function calls'''
input_directory_name = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-to-modify-csv/'

output_directory_name = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

rename_acc_batch_csv(input_directory_name, output_directory_name)