#!/usr/bin/env python
# coding: utf-8

# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# this function returns a dataframe and its transpose
def read_data(filePath, cols):
    """
    This function defines all the parameters for reading the excel file:
    filePath: the url of the file,
    cols: columns to be dropped in the dataframe
    """
    df = pd.read_excel(filePath, sheet_name='Data', skiprows=3)
    df = df.drop(cols, axis=1)
    df.set_index('Country Name', inplace=True)
    return df


filePath = 'https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=excel'
cols = ['Country Code', 'Indicator Name', 'Indicator Code']
df_CO2 = read_data(filePath, cols)


print(df_CO2)

# for clustering, Year 1990 and 2019 are extracted
data_CO2 = df_CO2.loc[:, ['1990','2019']]
print(data_CO2)

print(data_CO2.isnull().sum())

# drop null values
dataCO2 = data_CO2.dropna()

# create a scatterplot that shows the distribution of data points between 1990 and 2019
plt.figure(figsize=(14,12))
plt.scatter(dataCO2['1990'], dataCO2['2019'], color='r')
plt.title('Scatterplot of CO2 Emissions in the world between 1990 and 2019', fontsize=20)
plt.xlabel('Year 1990', fontsize=20)
plt.ylabel('Year 2019', fontsize=20)
plt.show()

# min-max normalization is created using a custom function
def scaler(df):
    """Accepts a dataframe and normalises the columns and return points between 0 and 1"""
    df_min = df.min()
    df_max = df.max()
    scaled = (df - df_min) / (df_max - df_min)
    return scaled

# a function is created which scales each column of the dataframe
def normalised_data(data):
    """"the dataframe is passed into the above function and it returns the scaled inputs of each column"""
    for col in data.columns:
        data[col] = scaler(data[col])
    return data

df_copy = dataCO2.copy()

norm_data = normalised_data(df_copy)

data_cluster = norm_data[['1990', '2019']].values
data_cluster

# the best number of clusters is chosen using sum of squared error
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=10, random_state=0)
    kmeans.fit(data_cluster) # the normalised data is fit using KMeans method
    sse.append(kmeans.inertia_)

    
plt.figure(figsize=(12,10))
plt.plot(range(1, 11), sse)
plt.xlabel('no of clusters', fontsize=15)
plt.ylabel('sse', fontsize=15)
plt.title('KMeans Elbow Method for CO2 emissions data', fontsize=18)
plt.show()

# the elbow method show 4 is the best number of clusters
# therefore it is used to cluster the data points using the KMeans function
kmeans = KMeans(n_clusters=4, random_state=0)
y_predict = kmeans.fit_predict(data_cluster)

# the cluster centroids are determined 
cluster_centroid = kmeans.cluster_centers_
print(cluster_centroid)

# a scatterplot is plot to visualize the clusters and the centroids
plt.figure(figsize=(14,12))
plt.scatter(data_cluster[y_predict == 0, 0], data_cluster[y_predict == 0, 1], s = 25, c = 'red', label='First Cluster')
plt.scatter(data_cluster[y_predict == 1, 0], data_cluster[y_predict == 1, 1], s = 25, c = 'blue', label='Second Cluster')
plt.scatter(data_cluster[y_predict == 2, 0], data_cluster[y_predict == 2, 1], s = 25, c = 'magenta', label='Third Cluster')
plt.scatter(data_cluster[y_predict == 3, 0], data_cluster[y_predict == 3, 1], s = 25, c = 'green', label='Fourth Cluster')
plt.scatter(cluster_centroid[:, 0], cluster_centroid[:,1], s = 200, c = 'black', marker='+', label = 'Cluster Centroids')
plt.title('Cluster and Centroids of countries between 1990 and 2019 in terms of CO2 emission', fontsize=20)
plt.xlabel('Year 1990', fontsize=20)
plt.ylabel('Year 2019', fontsize=20)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.show()

# the clusters are stored in a column called cluster
dataCO2['k_clusters'] = y_predict
print(dataCO2)


firstClusterData = dataCO2[dataCO2['k_clusters'] == 0]
secondClusterData = dataCO2[dataCO2['k_clusters'] == 1]
thirdClusterData = dataCO2[dataCO2['k_clusters'] == 2]
fourthClusterData = dataCO2[dataCO2['k_clusters'] == 3]

print(secondClusterData)

# create a figure and axis object
fig, ax = plt.subplots(figsize=(18,12))

# set the width of the bars
width = 0.35
xticklabels = ['East A&P', 'High Income', 'IBRD only', 'IDA & IBRD', 'L&M Income', 'LDD', 'Middle Income', 'OECD Mem', 'PDD', 'UMI']

# create a bar for the 1990 values
x = np.arange(len(secondClusterData.index))
rects1 = ax.bar(x - width/2, secondClusterData['1990'], width, label='1990')

# create a bar for the 2019 values
rects2 = ax.bar(x + width/2, secondClusterData['2019'], width, label='2019')

# set the x-tick labels to the defined list xticklabels
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)

# set the y-axis label
ax.set_ylabel('Kiloton', fontsize=20)
ax.set_title('CO2 emission(kt) represented as a grouped bar plot for the second cluster', fontsize=25)

# add a legend
ax.legend()

# show the plot
plt.show()

print(thirdClusterData)

# create a figure and axis object
fig, ax = plt.subplots(figsize=(16,12))

# plot the line chart
for i in thirdClusterData.index:
    ax.plot(thirdClusterData[['1990', '2019']].loc[i], label=i)

# set x and y-axis labels
ax.set_xlabel('Year', fontsize=15)
ax.set_ylabel('CO2 emission(kt)', fontsize=15)
ax.set_title('Multiple line plot showing the change in CO2 emission between 1990 and 2019', fontsize=20)

# add a legend
ax.legend(bbox_to_anchor=(1.0, 1.0))

# show the plot
plt.show()

# set the random seed for reproducibility
np.random.seed(140)

# randomly select 5 rows
random_rows = firstClusterData.sample(n=5)

# extract the numerical variable to plot 
values = random_rows['2019']

# create labels for the pie chart
labels = ['South Africa', 'Dominican Republic', 'South Asia (IDA & IBRD)', 'Latin America & Caribbean', 'Middle East & North Africa (excluding high income)']

# create a pie chart
fig, ax = plt.subplots(figsize=(12,10))
ax.pie(values, labels=labels, autopct='%1.1f%%')

# set the title
ax.set_title('Pie Chart of Randomly Selected Countries Using the First Cluster', fontsize=20)

# show the plot
plt.show()

# for the data fitting, the CO2 dataframe is transposed
CO2_transposed = df_CO2.transpose()
print(CO2_transposed)

# the rows of the transposed data is sliced for easier analysis
CO2_sliced = CO2_transposed.loc['1990':'2019',:]
CO2_sliced

# Column United states is extracted and a dataframe is created
df_US = pd.DataFrame({
    'Year' : CO2_sliced.index,
    'United States' : CO2_sliced['United States']
})
df_US.reset_index(drop=True)

# the year column is converted to integer
df_US['Year'] = np.asarray(df_US['Year'].astype(np.int64))

# a line plot of CO2 emission in USA
plt.figure(figsize=(12,10))
plt.plot(df_US['Year'], df_US['United States'])
plt.title('Time series plot showing change in CO2 emission in United States', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('kt CO2 emission', fontsize=15)
plt.show()

# this function returns a polynomial 
def polynomial(a, b, c, d):
    """
    Calculates a polynomial function which accepts:
    a: this is the years of the data
    b,c,d are the constants which define the equation
    """
    return b + c*a + d*a**2 
    

# the curve_fit function is used to fit the data points 
# and it accepts the polynomial function, years and United States Column as parameters
param, cov = curve_fit(polynomial, df_US['Year'], df_US['United States'])
print(f'Param: {param}')
print(f'Covariance: {cov}')

# the function calculates the confidence intervals
def err_ranges(x, y):
    """
    This function calculates the confidence intervals of the data points
    it returns the lower and upper limits of the data points
    """
    ci = 1.96 * np.std(y)/np.sqrt(len(x))
    lower = y - ci
    upper = y + ci
    return lower, upper

# an array of years ranging from 1990 to 2035 is created for forecast
year = np.arange(1990, 2035)
# the forecast for the next 14 years is calculated using the polynomial function
forecast = polynomial(year, *param)

# the lower and upper limits of the forecast are calculated using the err_ranges function
lower, upper = err_ranges(year, forecast)

plt.figure(figsize=(12,10))
plt.plot(df_US["Year"], df_US["United States"], label="CO2 emission")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, lower, upper, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("CO2 emission")
plt.legend()
plt.show()

# Here the forecast for the next 16 years are put in a dataframe 
df_forecast = pd.DataFrame({'Year': year, 'forecast': forecast})
forecast_fourteen_years = df_forecast.iloc[30:,]
print(forecast_fourteen_years)





