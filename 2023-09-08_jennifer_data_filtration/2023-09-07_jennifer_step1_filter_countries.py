"""
File_name: 2023-09-07_jennifer_step1_filer_countries.py
Author: Jennifer Fortuny I Zhan
Date: Thursday, 2023-09-07

This file contains the python code I used to extract the countries with the top 10 gdp per capita from the original dataset.

"""
# Make sure to work in an enviornment were I have pandas installed.
import pandas as pd

# We will first load the entire data set into a variable:
# Make sure I am in the same working directory as the data.
data = pd.read_csv('adult.data', header=None)

# Given that our file has no column names, I will add them by referencing the adult.names file.
# After view in the list using less on the terminal, I notice that the last few lines of the file are about the attributes.
# I know that the first attribute is "age", and all the lines after it, until the last line of the file, are attributes.
# To begin, I will find out the line number for "age: ", and use this to help extract the other attribute names later.
# First I open the file containing the attribute name in read mode, and I read its lines into a list.
with open('adult.names', 'r') as infile:
    lines = infile.readlines()

# Then, I look through this list from the end, to find where "age: " appears:
for index, line in enumerate(reversed(lines), start=1):
    if "age: " in line:
        last_n_lines_we_need = index
        break

print(f" 'age: ', and then all the other attributes, begin to show from the last {last_n_lines_we_need}th line of the file.")
# The result shows that age, and the other attributes, begin to apear from the last 14th line of the file.
# Now I will join these 14 lines into a list.
# The lines containing the attributes are from line -14 until the end of the file, I make this into a variable:
# However, I noticed I forgot a line that apeared 2 lines before the line containing age: which was also part of the columns in our data.
# This is the coloumn which indicates income, and it takes on the values <=50K and >50K.
# So I will increase the number for the last_n_lines_we_need by 1.
# And also add one to the line we begin our search from, because there is an empty line between <=50K, >50K and age:.
attribute_lines = lines[-(last_n_lines_we_need + 1) + 1:]

# Now, the attributes always come with a :, like in "age: ".
# So I look for that in each item on our list
# When the : is present, I split that item and take what ends up at the 0th index.
# This could be the word that comes between :
attributes = []
for line in attribute_lines:
    attribute_name = line.split(":")[0]
    attributes.append(attribute_name.strip())

# We also need to add a name for the column that indicates income.
# This column is describe as the first attribute in the adult.name file, but is the last column in the adult.data file.
# So I will add it to the end of our attributes list.
attributes.append('income')

# Now we have the attribute names in a list.
# We already loaded out data at the begining.
# Now we add the column names we just found to the data:
data.columns = attributes
print(data.columns)

# List of countries with top 10 GDP percapital we found previously.
# First I want to check how the countries are written in the file itself, to prevent mismatching.
uniq_country_names = [country.strip() for country in data['native-country'].unique()]
print(uniq_country_names)
# I create a list of the countries we are interested in, matching the country name format in the file:
top_10_countries = [
    'Luxembourg', 'Ireland', 'Norway', 'Switzerland', 'Singapore', 
    'Qatar', 'United-States', 'Iceland', 'Denmark', 'Australia'
]
# Before filtering the data, I want to make sure that these 10 countries exist in our original data.
for country in top_10_countries:
    if country not in uniq_country_names:
        print(f"'{country}' doesn't appear in the original data.")
    else:
        print(f"'{country}' is in our data")
# The results showed that most countries we were interested in were not in our list
# Therefore I will decide on a new set of countries to focus on.
# I decided to look at only South-East Asian countries that exist in the original data.
# They are of investigative interest because many South-East Asian economies are developing economies experiencing growth.
south_east_asia = ["Cambodia", "Laos", "Philippines", "Vietnam", "Thailand"]

# Finally, I filter our original data by these five countries.
# I noticed that the entries in the native-country column of the original data are surrounded by white space.
# Therefore I will remove these white spaces first.
data['native-country'] = data['native-country'].str.strip()

# The variable below only has the rows of data about people that have one of our five countries in their native-country column.
filtered_by_country = data[data['native-country'].isin(south_east_asia)]
print(filtered_by_country)

# I save this output as a csv file for further processing and sharing.
filtered_by_country.to_csv('2023-09-08_jennifer_filtered_by_country.csv', index=False)
# The csv file it produced contains 320 rows, and 15 columns.
# Next, I will reduce the number of columns to 6.
