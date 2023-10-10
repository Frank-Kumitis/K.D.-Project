import pandas as pd
import numpy as np

# Prior to running this code, the excel spreadsheet had to be converted to a csv using excel's save as csv prompt.
# If you do not go through this process, the file will not work as the encoded ASNI nature of the base file cannot be used with pandas

# Read csv and set columns
data = pd.read_csv('C:\\Users\\Evan Mitchell\\OneDrive\\Documents\\Knowledge Discovery and Data Mining\\Project\\oasis_longitudinal_demographics.csv')
data.columns = ['Subject ID', 'MRI ID',	'Group', 'Visit', 'MR Delay', 'M/F', 'Hand', 'Age', 'EDUC',	'SES',	'MMSE',	'CDR',	'eTIV',	'nWBV',	'ASF']

# Drop Subject ID and MRI ID
data = data.drop(['Subject ID'], axis=1)
data = data.drop(['MRI ID'], axis=1)
print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))
print(data.head())

# Replace missing values for attribute SES with median of attribute as values are only whole numbers and mean would not keep this trend
data2 = data['SES']
print('Before replacing missing values:')
print(data2)
data2 = data2.fillna(data2.median())
print('\nAfter replacing missing values:')
print(data2)

# Drop rows in data that are missing data
print('Number of rows in original data = %d' % (data.shape[0]))
data = data.dropna()
print('Number of rows after discarding missing values = %d' % (data.shape[0]))

# Drop rows that are duplicates
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
data2 = data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (data2.shape[0]))