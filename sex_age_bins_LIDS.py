'''IMPORTS'''
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from pyActigraphy.analysis import LIDS


# define a LIDS object
lids_obj = LIDS()

def initialize_user_dictionary(demographics_filename):
    """
    Takes in a csv file with information about each user ID (age, sex, etiology), and creates a dictionary with
    ID values as keys and tuple values (Age, Sex, and Etiology)
    :param demographics_filename
    :return age_sex_etiology_dict
    """
    # define the filepath
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files/'

    # read in the file
    dem_df = pd.read_csv(fpath + demographics_filename)

    # initialize an empty dictionary
    age_sex_etiology_dict = {}

    for index, row in dem_df.iterrows():

        ID = row['Actigraphy_File'].replace(".cwa", "")
        Age = row['Age']
        Sex = row['Sex']
        Etiology = row['Etiology']
        dict_values = (Age, Sex, Etiology)
        age_sex_etiology_dict[ID] = dict_values

    return age_sex_etiology_dict






'''FUNCTION CALLS'''
filename = 'AgeSexDx_n166_2023-07-13.csv'
initialize_user_dictionary(filename)
