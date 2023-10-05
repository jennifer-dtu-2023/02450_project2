"""
File_name: 2023-09-08_jennifer_step2_filter_columns.py
Author: Jennifer Fortuny I Zhan
Date: Thursday, 2023-09-08
Main edits: 2023-09-15, added "income" as a column".

This file contains the python code I used to filter 6 out of 15 columns fron the filtered_by_country.csv file.

"""
# I will only retain these coloumns: age, education-num, hour-per-week, workclass, and occupation.
# The first three columns are continuous attributes, the last two are categorical.
# I use less on the terminal to get a look at the data we are wokring on:
# we have all the headers in place, and no index column.

# As before, I make sure to work in an enviornment with the pandas package:
import pandas as pd

# First I load my filtered_countries.csv file
data = pd.read_csv('2023-09-08_jennifer_filtered_by_country.csv')

# Now I specify the columns that I want to keep:
keep_attributes = data[['age', 'education-num', 'hours-per-week', 'workclass', 'occupation', 'income']]

# Finally, I save this filtered data as a csv file for further processing and sharing.
keep_attributes.to_csv('2023-09-08_jennifer_filtered_complete.csv', index=False)

# Below is some information about this dataset:
rows, columns = keep_attributes.shape
datapoints = rows * columns
print(f"We have {rows} rows and {columns} columns, giving us {datapoints} datapoints.\n")
print(f"The {columns} columns, i.e. attributes, are:")
print(", ".join(keep_attributes.columns))

