import pyActigraphy
import os

def read_input_data(filename):
    """
    Reads in a SINGLE cwa file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # set a unique virtual environment
    fpath = os.environ["WINNEBECK_TEST_DATA"] # change the virtual environment, corresponding to the filepath you want

    #  read in the raw data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename, frequency="10Min", use_metadata_json=False) # frequency="10Min"

    return raw

#raw = read_input_data('1.csv.gz')
#print('IS: ', raw.IS())
#print("IV: ", raw.IV())



