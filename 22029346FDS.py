#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:31:17 2023

@author: cynthiaUdoye
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# page source:data6.csv

# read the CSV file using pandas for visualization
df = pd.read_csv('data6.csv')

# convert the dataframe to a numpy array
data = df.to_numpy()
print(data)

# define a function that creates a histogram and the arguments are passed into the defined function to create the histogram
def histogram(data, title, xlabel, ylabel):
    """
    This creates an histogram of the data and it accepts the following as parameters 
    data: This is the data to be plot into the histogram
    title: The title of the histogram
    xlabel: The label of the x-axis
    ylabel: The label of the y-axis
    """
    
# plot the histogram
plt.figure(figsize=(10,8))
plt.hist(data, bins=10)
plt.xlabel('Weight (kg)', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('Distribution of Newborn Weights', fontsize=20)

# calculate the average weight of newborns in the region
W = np.mean(data)
print('Average Weight of Newborns:', W)

# calculate the weight above which 10% of newborns are born
X = np.percentile(data, 90)
print('Weight Above 10%:', X)

# show and add the value of W on the histogram with dotted lines
plt.axvline(W, color='red', linestyle='dashed', linewidth=1, \
            label=f'Average Weight of Newborns (W) = {W:.2f}kg')

# show and add the value of X on the histogram with dotted lines
plt.axvline(X, color='green', linestyle='dashed', \
            linewidth=1, label=f'Weight Above 10%(X) = {X:.2f}kg')

# add legend and then show the graph
plt.legend(loc='upper left')
plt.show()
