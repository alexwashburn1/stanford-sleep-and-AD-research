import pyActigraphy
import os

def read_input_data(filename):
    """
    Reads in a SINGLE cwa file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    #fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/all-data-files/'

    fpath = os.environ["ACTIGRAPHY_DATA_FILES"]

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename, use_metadata_json=False)

    return raw



