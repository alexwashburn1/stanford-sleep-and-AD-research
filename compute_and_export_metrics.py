import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import os
from pyActigraphy.analysis import Cosinor  # for cosinor analysis
import csv

def read_input_data(filename):
    """
    Reads in a SINGLE csv.gz file for actigraphy analysis
    :param filename: the name of the file to read in
    :return: raw processed file
    """

    # Get the directory path of the current script or module
    fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

    # actually read in the data
    raw = pyActigraphy.io.read_raw_bba(fpath + filename)

    return raw

# guidelines: for now, focus on cosinor, IV/IS/RA, fragmentation (k), sleep regularity index, LIDS (potentially explore other methods)

def export_metrics_to_compare(filename): #TODO - add output_dir as a parameter eventually...
    """
    takes in a raw file for a subject, and generates a csv file with computed metrics for that subject.
    :param output_dir: the directory to write the output file to
    :param raw: the raw file object containing data to compute metrics for
    :return:
    """

    # read in the filename, convert to a raw object
    raw = read_input_data(filename)

    # initialize dictionary for metrics
    metrics_dict = {}

    # Add filename to the dictionary
    metrics_dict['filename'] = filename

    # Inter-daily stability (IS)
    IS = raw.IS()
    metrics_dict['IS'] = IS

    # Inter-daily variability (IV)
    IV = raw.IV()
    metrics_dict['IV'] = IV

    # Relative amplitude (RA)
    RA = raw.RA()
    metrics_dict['RA'] = RA

    # kAR (activity to rest probability)
    kAR = raw.kAR(0.1)  # todo - threshold  ?
    metrics_dict['kAR'] = kAR

    kRA = raw.kRA(0.1)
    print('RA attempt, ', kRA)  # TODO - create a histogram to determine an appropriate threshold ?

    # cosinor metrics
    cosinor = Cosinor()
    cosinor.fit_initial_params['Period'].value = 2880  # set period value, for 30-sec sampling frequency
    cosinor.fit_initial_params['Period'].vary = False  # do not vary the period
    cosinor.fit_initial_params.pretty_print()  # print the parameters in a neatly formatted table

    results = cosinor.fit(raw.data)
    cosinor_metrics_dict = results.params.valuesdict()

    # parse out dictionary values to individual variables
    metrics_dict['amplitude'] = cosinor_metrics_dict['Amplitude']
    metrics_dict['acrophase'] = cosinor_metrics_dict['Acrophase']
    metrics_dict['period'] = cosinor_metrics_dict['Period']
    metrics_dict['mesor'] = cosinor_metrics_dict['Mesor']

    metrics_dict['reduced chi^2'] = results.redchi
    metrics_dict['aic'] = results.aic
    metrics_dict['bic'] = results.bic

    print('metrics dict: ', metrics_dict)
    return metrics_dict

def compute_metrics_full_batch(input_dir, output_file):
    """
    iterates over all files in a directory, calculates the metrics using a helper function, and writes those
    metrics to a csv file.
    :return:
    """

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'IS', 'IV', 'RA', 'kAR', 'amplitude', 'acrophase',
                                                     'period', 'mesor', 'reduced chi^2', 'aic', 'bic'])
        writer.writeheader()

        for filename in os.listdir(input_dir):
            if filename.endswith(".csv.gz"):
                metrics_dict = export_metrics_to_compare(filename)
                writer.writerow(metrics_dict)


def create_empty_csv_file(file_path):
    """
    will generate an empty csv file to ultimately write metrics to for each file. each line represents one patient.
    :param file_path: the file path to write the file to.
    :return:
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(['filename', 'IS', 'IV', 'RA', 'kAR', 'amplitude', 'acrophase', 'period', 'mesor', 'reduced chi^2', 'aic', 'bic'])


'''FUNCTION CALLS'''
# initialize blank csv file
output_path = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/output_metrics.csv'

create_empty_csv_file(output_path)

# write all of the data to the csv file
input_dir = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/'

output_file = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/' \
                 'Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/stanford-modified-csv/output_metrics.csv'

#compute_metrics_full_batch(input_dir, output_file)

filename = '78203_0000000534-timeSeries.csv.gz'
export_metrics_to_compare(filename)

df = pd.read_csv(input_dir+filename)
print('df: ', df)
fig = df.hist(by = 'acc')
fig.show()



raw = read_input_data(filename)
print('type raw: ', type(raw))




