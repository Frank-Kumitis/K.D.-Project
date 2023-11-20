import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read csv and set columns
url = 'https://github.com/mitch172/K.D.-Project/blob/main/oasis_longitudinal_demographics.csv?raw=true'
data = pd.read_csv(url)
data.columns = ['Subject ID', 'MRI ID',	'Group', 'Visit', 'MR Delay', 'M/F', 'Hand', 'Age', 'EDUC',	'SES',	'MMSE',	'CDR',	'eTIV',	'nWBV',	'ASF']

# Drop Subject ID, MRI ID, and Hand
data = data.drop(['Subject ID'], axis=1)
data = data.drop(['MRI ID'], axis=1)
data = data.drop(['Hand'], axis=1)

# Change values in data from Nondemented to 0 and Demented to 1 for plotting purposes
def update_value(value):
    return 0 if 'Nondemented' in value else 1
data['Group'] = data['Group'].apply(lambda x: update_value(x))

# Display graph comparing MMSE to group assignment
counts = data['MMSE'].value_counts().sort_index()
counts.plot.bar('MMSE', 'Group')
plt.xlabel('MMSE value')
plt.ylabel('Total amount of Demented patients')
plt.title('Patient MMSE value compared to Demented status')
plt.show()

# Display graph comparing gender to group assignment
def update_value(value):
    return 0 if 'M' in value else 1
data['M/F'] = data['M/F'].apply(lambda x: update_value(x))
grouped_totals = data.groupby('Group')['M/F'].sum()
grouped_totals.plot.bar('M/F', 'Group')
plt.xlabel('Male (0) vs. Female (1)')
plt.ylabel('Total amount of Demented patients')
plt.title('Demented patients grouped by sex')
plt.show()

# Display graph comparing age to group assignment
filtered_data = data[data['Group'] == 1]
count_values = filtered_data['Age'].value_counts().sort_index()
count_values.plot.bar('Age', 'Group')
plt.xlabel('Age')
plt.ylabel('Total amount of Demented patients')
plt.title('Demented patients catagorized by age')
plt.show()

# Display boxplot examining if there are any outliers based on age
data['Age'].plot.box()
plt.xlabel(None)
plt.ylabel('Age')
plt.title('Ages for examing outliers')
plt.show()