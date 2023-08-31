import pyActigraphy
import os

def read_input_data(filename):
    """
    Reads in a SINGLE cwa file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # set a unique virtual environment
    fpath = os.environ["ACTIGRAPHY_DATA_FILES"]

    #  read in the raw data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename, use_metadata_json=False)

    return raw



