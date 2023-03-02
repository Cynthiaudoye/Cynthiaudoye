#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:03:57 2023

@author: CynthiaUdoye
"""
# import standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# page source: https://www.kaggle.com/datasets/thedevastator/cryptocurrency-price-market-data?select = coin_gecko_2022-03-16.csv
# read the csv file for visualization
crypto_data = pd.read_csv('coin_gecko_2022-03-16.csv')
print(crypto_data)
print(crypto_data.head())

# check for the sum of the null values in each column
print(crypto_data.isnull().sum())

# drop null values
data_crypto = crypto_data.dropna()
data_crypto.reset_index(drop = True)
print(data_crypto.describe())

# for my multiple plot i want to use the first 10 rows of the data
crypto_multiplot = data_crypto.iloc[:10,[1, 3, 4, 5]]
print(crypto_multiplot)

# define a function that creates a multiple plot
def multiplot(x_data, y_data, xlabel, ylabel, labels, title, color):
    """
    This creates a multiple line plot and accepts the following as parameters 
    
    x_data : This is the column to be plot on the x-axis
    y_data : These are the columns to be plot on the y-axis
    xlabel : This is the label of the x-axis
    ylabel : This is the label of the y-axis
    labels : These are the labels of each data on the y-axis
    title : This is the name\title of the plot
    color : These are the colors of the line plots
    """
    
    plt.figure(figsize = (16, 12), dpi = 200)
    plt.title(title, fontsize = 20)
    for i in range(len(y_data)):
        plt.plot(x_data, y_data[i], label = labels[i], color = color[i])
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.legend()
    plt.show()
    return

# the argument are passed into the function to display the multiple line plots
x_data = crypto_multiplot['symbol']
y_data = [crypto_multiplot['1h'], crypto_multiplot['24h'], crypto_multiplot['7d']]
xlabel = 'Cryptocurrency symbols'
ylabel = 'Change trends'
labels = ['1 hour', '24 hours', '7 days']
title = 'A multiple line plot showing the trends of cryptocurrencies across different time frame'
color = ['blue', 'red', 'green']

print(multiplot(x_data, y_data, xlabel, ylabel, labels, title, color))

# define a function that creates a histogram subplot
def histogram(data, title, label, color, xlabel):
    """
    This creates an histogram of each data columns using a subplot
    
    data : This is the data to be plot into the histograms
    title : The title of each subplots
    label : The labels of each subplots
    color : The color of each subplots
    xlabel : The label of the x-axis
    """
    
    plt.figure(figsize = (12, 8), dpi = 200)
    for i in range(len(data)):
        plt.subplot(1, 3, i+1).set_title(title[i])
        plt.hist(data[i], color = color[i], label = label[i])
        plt.xlabel(xlabel)
        plt.legend()
    plt.show()
    return
        
# arguments are passed into the defined function to create the histogram subplots
data = [data_crypto['1h'], data_crypto['24h'], data_crypto['7d']]
title = ['One hour', '24 hours', '7 days']
color = ['magenta', 'purple', 'yellow']
xlabel = 'trading percentage'
    
print(histogram(data, title, xlabel, color, xlabel))


# a random sample is generated for 10 samples and 
data_bar = data_crypto.sample(n = 10, random_state = 1)
print(data_bar)

# define a function that visualizes a sub-bar plot
def barplot(data, label_list, width, ylabel, xlabel, label, title, color):
    """
    This creates a barplot of each columns using subplot function and it accepts the following as parameters
    
    data : The columns of the data to be plot
    label_list : The label of the points on the x-axis
    width : width of each bars
    ylabel : The label of the y-axis
    xlabel : The label of the x-axis
    label : The label of the each subplots
    title : The title\name of each subplots
    color : The colors of each subplots
    """

    x =np.arange(len(label_list))
    plt.figure(figsize = (12, 6), dpi = 200)
    for i in range (len(data)):
        plt.subplot(1, 2, i + 1).set_title(title[i], fontsize = 6, fontweight = 'bold')
        plt.bar(x-width, data[i], width, label = label[i], color = color[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel[i])
        plt.xticks(x-width, label_list, rotation = 'vertical')
        plt.legend()
    plt.show()
    return
    
    
# the arguments below are passed into the defined function to create a subplot of barplots
data = [data_bar['24h_volume'], data_bar['mkt_cap']]
label_list = data_bar['symbol']
width = 0.4
ylabel = ['A day trading volume', 'Market Capitalization']
xlabel = 'Crypto symbols'
label = ['24 hour vol', 'Market Capitalization']
title = ['A bar plot of the 24 hour trading volume of 10 cryptocurrencies', 'A bar plot of the Market Capitalization of 10 cryptocurrencies']
color = ['cyan', 'pink']   
print(barplot(data, label_list, width, ylabel, xlabel, label, title, color))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    