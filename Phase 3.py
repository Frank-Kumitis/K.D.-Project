import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns


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

# Display graph comparing Gender to group assignment
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

# Display scatterplot examining age vs MMSE
data.plot.scatter(x = 'Age', y = 'MMSE')
plt.show()

# Display scatterplot examining age vs MMSE
data.plot.scatter(x = 'Age', y = 'EDUC')
plt.show()

# Display scatterplot examining age vs MMSE
data.plot.scatter(x = 'Age', y = 'ASF')
plt.show()

# Preserving data for future use
data2 = data

# Make decision tree for finding Demented status
data = data.drop(['Visit'], axis=1)
data = data.drop(['MR Delay'], axis=1)
data = data.drop(['M/F'], axis=1)
X = data.drop('Group', axis=1)
y = data['Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
plt.figure(figsize=(30, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y.unique().astype(str))
plt.show()

#Correlation
correlation_matrix = data.corr()
plt.matshow(correlation_matrix)
correlation_matrix.convert_dtypes(str)
plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns)
plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns)
plt.colorbar()
plt.show()

# Principal Component Analysis

# Refreshing dropped columns
data = data2


features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'M/F', 'CDR']
data.dropna(inplace=True)

x = data.loc[:,features].values
y = data.loc[:,['Group']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_data = pd.DataFrame(data = principal_components,
                              columns=['principal component 1', 'principal component 2'])

finished_data = pd.concat([principal_data, data[['Group']]], axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('principal component 1')
ax.set_ylabel('principal component 2')
ax.set_title('2 component PCA')

targets = [0,1]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finished_data['Group'] == target
    ax.scatter(finished_data.loc[indicesToKeep, 'principal component 1']
               , finished_data.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


#Clustering

km_data = finished_data.dropna()

kmeans = KMeans(n_clusters=3, n_init='auto', random_state=50)
pd.set_option('mode.chained_assignment', None)
kmeans.fit(km_data)
km_data['Clusters']= kmeans.labels_

sns.scatterplot(x='principal component 1',
                y='principal component 2',
                hue='Clusters', data=km_data)

plt.title('K-Means on Principal Components')
plt.show()