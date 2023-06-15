''' IMPORTS AND INPUT DATA '''
import pyActigraphy
from pyActigraphy.analysis import SSA
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import os

# Get the directory path of the current script or module
fpath = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/'

# actually read in the data
raw = pyActigraphy.io.read_raw_bba(fpath+'79036_0000000504-timeSeries.csv.gz')

# verify that this was done correctly
print(raw.name)
print(raw.start_time)





