'''IMPORTS'''
import pandas as pd
import numpy as np
import csv


def initialize_user_dictionary(demographics_filename):

    # define the filepath
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

    # read in the file
    dem_df = pd.read_csv(fpath + demographics_filename)

    # initialize an empty dictionary
    age_sex_etiology_dict = {}

    for index, row in dem_df.iterrows():
        
        ID = row['Actigraphy_File'].replace(".cwa", "")
        print('ID: ', ID)


'''FUNCTION CALLS'''
filename = 'AgeSexDx_n166_2023-07-13.csv'
initialize_user_dictionary(filename)
