import os
import csv

import os
import csv
# import pandas as pd
from datetime import datetime
from datetime import timedelta
import pyexcel_ods3
from dateutil import parser
from datetime import timedelta
from datetime import time



import pandas as pd
import csv
import pyexcel_ods3
from dateutil import parser

# def create_sleep_diary_ods(original_csv_file):
#     df = pd.read_csv(original_csv_file, delimiter=',')  # Update delimiter if necessary
#
#     for _, row in df.iterrows():
#         subject_id = row['ID']
#         new_ods_file = f"/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/{subject_id}.ods"
#
#         sheet_data = [["SubjectID", subject_id]]  # Add extra row with SubjectID
#         sheet_data.append([])  # Add blank row
#
#         sheet_data.append(["TYPE", "START", "END"])  # Write the header row
#
#         for i in range(1, 14):
#             date = row[f'D{i}_date']
#             wakeup = row[f'D{i+1}_wakeup']
#             onset = row[f'D{i}_onset']
#
#             if pd.notnull(date) and pd.notnull(wakeup) and pd.notnull(onset) and date != '########' and wakeup != '########' and onset != '########':
#                 date_format = '%d/%m/%y' if len(date.split('/')[2]) == 2 else '%d/%m/%Y'
#
#                 if len(date.split('/')[2]) == 2:
#                     date = date[:-2] + '20' + date[-2:]  # Assuming all years are in the 2000s
#
#                 onset_time = parser.parse(onset).strftime('%H:%M')
#                 start = parser.parse(date, dayfirst=True).strftime('%Y-%m-%d') + ' ' + onset_time
#
#                 wakeup_time = parser.parse(wakeup).strftime('%H:%M')
#                 end = parser.parse(row[f'D{i+1}_date'], dayfirst=True).strftime('%Y-%m-%d') + ' ' + wakeup_time
#
#                 sheet_data.append(['NIGHT', start, end])
#
#         # Save as .ods file
#         pyexcel_ods3.save_data(new_ods_file, {"Sheet 1": sheet_data})



def create_sleep_diary_ods(original_csv_file):
    df = pd.read_csv(original_csv_file, delimiter=',')

    for _, row in df.iterrows():
        subject_id = row['ID']
        new_ods_file = f"/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/{subject_id}.ods"

        sheet_data = [["SubjectID", subject_id]]
        sheet_data.append([])
        sheet_data.append(["TYPE", "START", "END"])

        for i in range(1, 14):
            date = row[f'D{i}_date']
            wakeup = row[f'D{i+1}_wakeup']
            onset = row[f'D{i}_onset']

            if pd.notnull(date) and pd.notnull(wakeup) and pd.notnull(onset) and date != '########' and wakeup != '########' and onset != '########':
                date_format = '%d/%m/%y' if len(date.split('/')[2]) == 2 else '%d/%m/%Y'

                if len(date.split('/')[2]) == 2:
                    date = date[:-2] + '20' + date[-2:]

                onset_time = parser.parse(onset).time()
                start = parser.parse(date, dayfirst=True)

                if time(0, 0) <= onset_time < time(8, 0):
                    # Advance the start date by one day if the onset time is at or beyond 00:00:00 am but before 8:00:00 am
                    start += timedelta(days=1)

                start = start.strftime('%Y-%m-%d') + ' ' + onset_time.strftime('%H:%M')

                wakeup_time = parser.parse(wakeup).strftime('%H:%M')
                end = parser.parse(row[f'D{i+1}_date'], dayfirst=True).strftime('%Y-%m-%d') + ' ' + wakeup_time

                sheet_data.append(['NIGHT', start, end])

        # Save as .ods file
        pyexcel_ods3.save_data(new_ods_file, {"Sheet 1": sheet_data})

# Usage example
original_csv_file = "/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/original_sleep_logs.csv"

create_sleep_diary_ods(original_csv_file)
