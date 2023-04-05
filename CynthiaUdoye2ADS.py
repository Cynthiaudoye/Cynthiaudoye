#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:28:18 2023

@author: CynthiaUdoye2ADS
"""

# import standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# page source: https://data.worldbank.org/topic/climate-change. select=API_19_DS2_en_csv_v2_5361599.csv
# create dataframes that takes a filename as argument
def create_dataframes(filename):
    # read the csv file from the World Bank website into pandas DataFrame with years as columns for visualization
    df_years = pd.read_csv(filename, skiprows=4)
    
    # Transpose the dataframe to get countries as columns
    df_transpose = pd.DataFrame.transpose(df_years)
    
     # drop unnecessary columns
    df_transpose = df_transpose.drop(['Country Code','Indicator Code','Unnamed: 66'], axis=0)
    
    # rename column
    df_transpose = df_transpose.rename({'Country Name': 'Country'})
    
     # remove the first row 
    df_transpose.columns = df_transpose.iloc[0]
    df_countries = df_transpose.iloc[1:]

    # return the dataframes
    return df_years, df_countries

df_years, df_countries = create_dataframes('Climate-change-data.csv')
# the untranspose dataframe
print(df_years.head())

# The transposed cleaned dataframe 
print(df_countries.head())

# selecting columns i need for the analysis.
df = df_years.drop(['Country Code','Indicator Code','Unnamed: 66'], axis=1)
print(df)


#for the indicators i want to select 4 
indicators= [ 'Urban population','CO2 emissions (kt)', 'Electric power consumption (kWh per capita)', 'Energy use (kg of oil equivalent per capita)']

# subsetting the indicators    
indi_df= df[df['Indicator Name'].isin (indicators)]

print(indi_df.head())

# creating the transpose of the dataframe indicators
indi_T =  pd.DataFrame.transpose(indi_df)

# Making Country names the columns of the dataset.
indi_T.columns = indi_T.iloc[0]
indi_T = indi_T.iloc[1:]
print(indi_T)

# Selecting two countries from different region of the world
countries = indi_T.loc[:, ['Argentina','United States','Malaysia','Germany','China','Nigeria' ]]
data_df = countries.iloc[1:,:]
print(data_df.head())


# dropping missing values from my dataset
data_df.dropna(inplace = True)
print(data_df)

# converting dataframe data type to numeric(float64)
new_df = data_df.apply(pd.to_numeric)
print(new_df)


# Generating a subset of new_df dataframe pertaining to urban population for all the chosen countries and using all rows and 6 columns 
urb_pop = new_df.iloc[:,[0,4,8,12,16,20]]

print(urb_pop)


# exploring statistical properties of the urban population indicators by getting the summary statistics
urb_pop.describe()

urb_pop.corr()

# still converting the index to numeric
urb_pop.index = pd.to_numeric(urb_pop.index)

print(urb_pop)


#define a function that creates a multiple plot
def multiplot(data_x, data_y, label, xlabel, ylabel, title):
    """
    This creates a multiple line plot and accepts the following as parameters
    xlabel: This is the label of the x-axis
    ylabel: This is the label of the y-axis
    title: This is the name/title of the plot
    labels: These are the labels of each data on the y-axis
    """
    plt.figure(figsize=(16,12), dpi=100)
    for i in range(len(data_y)):
        plt.plot(data_x, data_y[i], label=label[i])
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend()
    plt.show()
    return

# the arguments are passed into the function to display the multiple line plots
data_x = urb_pop.index
data_y = [urb_pop['Argentina'], urb_pop['United States'], urb_pop['Malaysia'], urb_pop['Germany'], urb_pop['China'], urb_pop['Nigeria']]
label = ['Argentina', 'US', 'Malaysia', 'Germany', 'China', 'Nigeria']
xlabel = 'Years'
ylabel = 'Urban population'
title = 'Trends of the Urban population across selected years'

multiplot(data_x, data_y, label, xlabel, ylabel, title)


# generating a subset of new_df dataframe pertaining to carbondioxide emissions for all the chosen countries 
CO2 = new_df.iloc[:,[1,5,9,13,17,21]]

# getting the index and converting to numeric
CO2.index = pd.to_numeric(CO2.index)

# getting the transpose
CO2_T = CO2.transpose()

# subsetting the transpose
CO2_T = CO2_T.loc[:,[2000,2003,2006,2009,2012]]
print(CO2_T)

CO2_T.iloc[:,0]


# define a function that visualizes a sub-bar plot 
def barplot(data, label_list, width, ylabel, xlabel, label, title, color):
    """
    This creates a barplot of each columns using subplot function
    and it accepts the following as parameters
    data: the columns of the data to be plot
    label_list: the label of the points on the x-axis
    width: width of each bars
    ylabel: the label of the y-axis
    xlabel: the label of the x-axis
    label: the label of the each subplots
    title: the title/name of each subplots
    color: the colors of each subplots
    """
    x = np.arange(len(label_list))
    fig, ax  = plt.subplots(figsize=(16,12))
  
    plt.bar(x - width, data[0], width, label=label[0], color=color[0])
    plt.bar(x, data[1], width, label=label[1], color=color[1])
    plt.bar(x + width, data[2], width, label=label[2], color=color[2])
    plt.bar(x + width*2, data[3], width, label=label[3], color=color[3])
    plt.bar(x + width*3, data[4], width, label=label[4], color=color[4])
    plt.title(title, fontsize=20)
    plt.xlabel(None)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(x + width, label_list, rotation='vertical')
    plt.legend()
    
    plt.show()


# the arguments below are passed into the defined function to create a subplot of barplots
data = [CO2_T.iloc[:,0], CO2_T.iloc[:,1], CO2_T.iloc[:,2], CO2_T.iloc[:,3], CO2_T.iloc[:,4]]
label_list = ['Argentina', 'US', 'Malaysia', 'Germany', 'China', 'Nigeria']
width = 0.15
ylabel = 'CO2 Emission'
xlabel = 'Countries'
label = ['2000', '2003', '2006', '2009', '2012']
title = 'Grouped bar chart of CO2 emission for various nations over years'
color = ['cyan', 'pink', 'blue', 'green', 'purple']
barplot(data, label_list, width, ylabel, xlabel, label, title, color)

# generating the dataframe of United States
Uni_sta = indi_T.loc[:,'United States']
print(Uni_sta)

# subsetting the columns
Uni_sta.columns = Uni_sta.iloc[0]
Uni_sta = Uni_sta[1:]

# droping null values
Uni_sta.dropna(inplace= True)
Uni_sta = Uni_sta.apply(pd.to_numeric) 
print(Uni_sta)

# exploring the correlation between varaibles for United State 
Uni_sta_cor = Uni_sta.corr().round(2)
print(Uni_sta_cor)

#plotting the heatmap and specifying the plot parameters
plt.imshow(Uni_sta_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(Uni_sta_cor)), Uni_sta_cor.columns, rotation=90)
plt.yticks(range(len(Uni_sta_cor)), Uni_sta_cor.columns)
plt.gcf().set_size_inches(11,9)

#labelling of the little boxes and creation of a legend
labels = Uni_sta_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x,y, '{:.2f}'.format(labels[y,x]), ha='center', va='center',
                  color='black')
plt.title('Correlation Map for United States')
plt.show()
plt.savefig("Heat Map of United States Region" )


# generating the dataframe of United States
Nigeria = indi_T.loc[:,'Nigeria']

# subsetting the columns
Nigeria.columns = Nigeria.iloc[0]
Nigeria = Nigeria[1:]

# droping null values
Nigeria.dropna(inplace= True)
Nigeria = Nigeria.apply(pd.to_numeric) 

# exploring the correlation between varaibles for United State 
Nigeria_cor = Nigeria.corr().round(2)
print(Nigeria_cor)

#plotting the heatmap and specifying the plot parameters
plt.imshow(Nigeria_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(Nigeria_cor)), Nigeria_cor.columns, rotation=90)
plt.yticks(range(len(Nigeria_cor)), Nigeria_cor.columns)
plt.gcf().set_size_inches(11,9)

#labelling of the little boxes and creation of a legend
labels = Nigeria_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x,y, '{:.2f}'.format(labels[y,x]), ha='center', va='center',
                  color='black')
plt.title('Correlation Map for Nigeria')
plt.show()
plt.savefig("Heat Map of Nigeria" )





