import os
import csv

import os
import csv
# import pandas as pd
from datetime import datetime


# def create_sleep_diary_csv(original_csv_file):
#     df = pd.read_csv(original_csv_file, delimiter=',')  # Update delimiter if necessary
#
#     for _, row in df.iterrows():
#         subject_id = row['ID']
#         new_csv_file = f"{subject_id}.csv"
#
#         with open(new_csv_file, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["SubjectID", subject_id])  # Add extra row with SubjectID
#
#             writer.writerow(["Type", "Start", "End"])  # Write the header row
#
#             for i in range(1, 15):
#                 date = row[f'D{i}_date']
#                 wakeup = row[f'D{i}_wakeup']
#                 onset = row[f'D{i}_onset']
#
#                 if pd.notnull(date) and pd.notnull(wakeup) and pd.notnull(onset) and date != '########' and wakeup != '########' and onset != '########':
#                     if len(date.split('/')[2]) == 2:
#                         date_format = '%d/%m/%y'
#                     else:
#                         date_format = '%d/%m/%Y'
#
#                     # Parse wakeup and onset times
#                     wakeup_time = datetime.strptime(wakeup, '%H:%M:%S').time().strftime('%H:%M')
#                     onset_time = datetime.strptime(onset, '%H:%M:%S').time().strftime('%H:%M')
#
#                     start = pd.to_datetime(date, format=date_format).strftime('%Y-%m-%d') + ' ' + onset_time
#                     end = pd.to_datetime(date, format=date_format).strftime('%Y-%m-%d') + ' ' + wakeup_time
#
#                     writer.writerow(['Night', start, end])
#
# # Usage example
# original_csv_file = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/original_sleep_logs.csv'
#
# #create_sleep_diary_csv(original_csv_file)

###### TRY ODS FILE TYPE ###### - working, but old
import pandas as pd
import csv
from pyexcel_ods3 import save_data

def create_sleep_diary_ods(original_csv_file):
    df = pd.read_csv(original_csv_file, delimiter=',')  # Update delimiter if necessary

    for _, row in df.iterrows():
        subject_id = row['ID']
        new_ods_file = f"{subject_id}.ods"

        data = [['SubjectID', subject_id], ['TYPE', 'START', 'END']]  # Initialize data with header rows

        for i in range(1, 15):
            date = row[f'D{i}_date']
            wakeup = row[f'D{i}_wakeup']
            onset = row[f'D{i}_onset']

            if pd.notnull(date) and pd.notnull(wakeup) and pd.notnull(onset) and date != '########' and wakeup != '########' and onset != '########':
                if len(date.split('/')[2]) == 2:
                    date_format = '%d/%m/%y'
                else:
                    date_format = '%d/%m/%Y'

                # Parse wakeup and onset times
                wakeup_time = datetime.strptime(wakeup, '%H:%M:%S').time().strftime('%H:%M')
                onset_time = datetime.strptime(onset, '%H:%M:%S').time().strftime('%H:%M')

                start = pd.to_datetime(date, format=date_format).strftime('%Y-%m-%d') + ' ' + onset_time
                end = pd.to_datetime(date, format=date_format).strftime('%Y-%m-%d') + ' ' + wakeup_time

                data.append(['NIGHT', start, end])

        save_data(new_ods_file, {'Sheet 1': data})

# Usage example
#original_csv_file = '/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/original_sleep_logs.csv'

#create_sleep_diary_ods(original_csv_file)


import pandas as pd
import csv
import pyexcel_ods3
from dateutil import parser

def create_sleep_diary_ods(original_csv_file):
    df = pd.read_csv(original_csv_file, delimiter=',')  # Update delimiter if necessary

    for _, row in df.iterrows():
        subject_id = row['ID']
        new_ods_file = f"/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/{subject_id}.ods"

        sheet_data = [["SubjectID", subject_id]]  # Add extra row with SubjectID
        sheet_data.append([])  # Add blank row

        sheet_data.append(["TYPE", "START", "END"])  # Write the header row

        for i in range(1, 14):
            date = row[f'D{i}_date']
            wakeup = row[f'D{i+1}_wakeup']
            onset = row[f'D{i}_onset']

            if pd.notnull(date) and pd.notnull(wakeup) and pd.notnull(onset) and date != '########' and wakeup != '########' and onset != '########':
                date_format = '%d/%m/%y' if len(date.split('/')[2]) == 2 else '%d/%m/%Y'

                if len(date.split('/')[2]) == 2:
                    date = date[:-2] + '20' + date[-2:]  # Assuming all years are in the 2000s

                onset_time = parser.parse(onset).strftime('%H:%M')
                start = parser.parse(date, dayfirst=True).strftime('%Y-%m-%d') + ' ' + onset_time

                wakeup_time = parser.parse(wakeup).strftime('%H:%M')
                end = parser.parse(row[f'D{i+1}_date'], dayfirst=True).strftime('%Y-%m-%d') + ' ' + wakeup_time

                sheet_data.append(['NIGHT', start, end])

        # Save as .ods file
        pyexcel_ods3.save_data(new_ods_file, {"Sheet 1": sheet_data})

# Usage example
original_csv_file = "/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/timeSeries-actigraphy-csv-files/sleep-journals/original_sleep_logs.csv"

create_sleep_diary_ods(original_csv_file)















