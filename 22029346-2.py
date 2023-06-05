#!/usr/bin/env python
# coding: utf-8

# ## Overview
# ### This notebook analyzes the trends in CO2 emissions by country using the OWID CO2 dataset. The analysis includes visualizations of CO2 emissions over time, a panel widget to navigate and display data for different years, and radio button widgets to switch between different measures of CO2. Additionally, it includes various visualizations of CO2 emissions by country, including bar plots, heat maps, and line plots. The analysis focuses on ten countries, including the United Kingdom, United States, China, Qatar, Russia, Germany, Japan, France, Canada, and Brazil. The notebook also explores the relationship between CO2 emissions and GDP and population, using scatter plots and regression lines.

# The CO2 emission dataset contains data on CO2 emissions from different countries over time. You can find the dataset on the Our World in Data website at the following link: https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv
# The dataset includes some variables such as:
# 
# iso_code: ISO 3166-1 alpha-3 country code
# country: Country name
# year: Year of observation
# co2: Annual CO2 emissions in metric tons
# co2_growth_prct: Percentage change in annual CO2 emissions
# co2_per_capita: Annual CO2 emissions per capita in metric tons
# share_global_co2: Share of global CO2 emissions in the given year
# cumulative_co2: Cumulative CO2 emissions since 1751 in metric tons
# cumulative_co2_per_capita: Cumulative CO2 emissions per capita since 1751 in metric tons
# share_global_cumulative_co2: Share of global cumulative CO2 emissions in the given year, etc
# This is a large dataset with observations for many countries over a long time period. 


#Load pandas library and necessary modules

from PIL import Image
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panel as pn 

#interactive tables that allows one interact with a table
pn.extension('tabulator') 

# interactive dataframes
import hvplot.pandas 


#Loading the dataset locally on the computer
df = pd.read_csv('co2emission.csv')


# Setting up cache to reduce the time required to retrieve data from its original source.
#if 'data' not in pn.state.cache.keys():
    #df=  pd.read_csv('https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv')
    #pn.state.cache['data'] = df.copy()
    
#else:
    #df = pn.state.cache['data']


#This inspects the dataset and the head.
#returns the first n rows of a DataFrame. By default, it returns the first 5 rows,
#However you can specify the number of rows you want to view using the n parameter.
df.head(5)


#Returns a random sample of rows from a DataFrame. By default, it returns a single row.
df.sample(10)


#Checking the column names
print(df.columns)


#Checking the sum of missing values
#Missing values vary for different countries
df.isna().sum()


# Fill missing values in the DataFrame with 0 and modify the DataFrame in-place
df.fillna(0, inplace=True)

# Display the first 5 rows of the modified DataFrame
df.head(5)


#Checking the sum of missing values again
df.isna().sum()


# ## Exploring the Nigerian CO2 Emission Data (Random Check)


# Load the dataset
df = pd.read_csv('co2emission.csv')

# Filter the dataframe to only include Nigeria's data
nigeria_df = df[df['iso_code'] == 'NGA']

# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
nigeria_df = nigeria_df[cols]

# Remove rows with missing data
nigeria_df = nigeria_df.dropna()

# Convert the 'year' column to datetime format
nigeria_df['year'] = pd.to_datetime(nigeria_df['year'], format='%Y')

# Set the 'year' column as the index
nigeria_df = nigeria_df.set_index('year')

# Plot the data as a line plot
plt.plot(nigeria_df)
plt.title('CO2 Emissions per Capita in Nigeria')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.show()


# ### The graph above shows the CO2 emissions per capita in Nigeria over time. From the graph, we can see that Nigeria's CO2 emissions per capita have been increasing over time, with some fluctuations. The highest peak in emissions occurred around the early 1980s, which could be attributed to the oil boom that occurred in Nigeria during that period. The graph also shows a slight decrease in emissions in recent years, which could be attributed to efforts by the Nigerian government to reduce emissions and adopt cleaner energy sources. Overall, the graph indicates that Nigeria, like many other developing countries, is facing a challenge in balancing economic growth with sustainable development.


# The summary statistics of the data
summary_stats = nigeria_df.describe()

# Print the summary statistics
print(summary_stats)


# ## CO2 emission for the last 10 years( Nigeria)


# Filter the dataframe to only include Nigeria's data
nigeria_df = df[df['iso_code'] == 'NGA']

# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
nigeria_df = nigeria_df[cols]

# Remove rows with missing data
nigeria_df = nigeria_df.dropna()

# Convert the 'year' column to datetime format and set it as the index
nigeria_df['year'] = pd.to_datetime(nigeria_df['year'], format='%Y')
nigeria_df = nigeria_df.set_index('year')

# Select the last 10 years of data
nigeria_last_10 = nigeria_df.loc['2011':'2020']

# Calculate the total CO2 emissions for each year
total_co2 = nigeria_last_10['co2_per_capita'].sum()

# Calculate the percentage of CO2 emissions for each year
co2_pct = nigeria_last_10['co2_per_capita'] / total_co2 * 100

# Create a pie chart
labels = nigeria_last_10.index.year.astype(str)
plt.pie(co2_pct, labels=labels, autopct='%1.1f%%')
plt.title('CO2 Emissions in Nigeria (Last 10 Years)')
plt.show()


# ### The pie chart shows the percentage contribution of CO2 emissions per capita in Nigeria for the last 10 years. From the chart, we can infer that the majority of CO2 emissions in Nigeria over the last 10 years were contributed by the years 2011 and 2014, which contributed to 12.3% and 10.8% of the total CO2 emissions, respectively. This indicates that efforts to reduce carbon emissions in Nigeria may need to focus on these years in particular. Additionally, the chart shows a slight decrease in CO2 emissions in 2020, which may be attributed to the COVID-19 pandemic and subsequent lockdown measures.

# ## Nigeria CO2 emission data in a Bar Chart


#Instantiate colour list for bar chart
colors = ['darkorange', 'steelblue', 'forestgreen', 'crimson', 'mediumvioletred', 'gold', 'cornflowerblue', 'indigo', 'salmon', 'limegreen']

# Filter the dataframe to only include Nigeria's data
nigeria_df = df[df['iso_code'] == 'NGA']

# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
nigeria_df = nigeria_df[cols]

# Remove rows with missing data
nigeria_df = nigeria_df.dropna()

# Convert the 'year' column to datetime format
nigeria_df['year'] = pd.to_datetime(nigeria_df['year'], format='%Y')

# Filter the dataframe to only include the last 10 years of data
last_10_years = nigeria_df['year'] > pd.Timestamp('now') - pd.DateOffset(years=10)
nigeria_df = nigeria_df[last_10_years]

# Set the 'year' column as the index
nigeria_df = nigeria_df.set_index('year')

# Create a bar chart of the CO2 emissions in the last 10 years
plt.bar(nigeria_df.index.year, nigeria_df['co2_per_capita'], color= colors)
plt.title('CO2 Emissions per Capita in Nigeria (Last 10 Years)', fontsize = 12)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize = 12)
plt.show()


# ### There has been a general increase in CO2 emissions per capita in Nigeria over the last decade. The year 2020 saw a slight drop(1% - see pie chart) in CO2 emissions per capita compared to the previous year. This may be due to the COVID-19 pandemic and the resulting decrease in economic activity. There is a lot of year-to-year variation in CO2 emissions per capita in Nigeria, suggesting that there are factors that affect emissions levels beyond just a linear trend.

# ## Total CO2 emissions for UK (Pie Chart Visualisation)


# Filter the dataframe to only include UK's data
uk_df = df[df['iso_code'] == 'GBR']

# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
uk_df = uk_df[cols]

# Remove rows with missing data
uk_df = uk_df.dropna()

# Convert the 'year' column to datetime format and set it as the index
uk_df['year'] = pd.to_datetime(uk_df['year'], format='%Y')
uk_df = uk_df.set_index('year')

# Select the last 10 years of data
uk_last_10 = uk_df.loc['2011':'2020']

# Calculate the total CO2 emissions for each year
total_co2 = uk_last_10['co2_per_capita'].sum()

# Calculate the percentage of CO2 emissions for each year
co2_pct = uk_last_10['co2_per_capita'] / total_co2 * 100

# Create a pie chart
labels = uk_last_10.index.year.astype(str)
plt.pie(co2_pct, labels=labels, autopct='%1.1f%%')
plt.title('CO2 Emissions in the UK (Last 10 Years)')
plt.show()


# ### From the pie chart visualization, we can see the percentage of CO2 emissions for each year in the UK from 2011 to 2020. The chart shows that in most of the years, the CO2 emissions remained fairly consistent, with the highest percentage of emissions occurring in 2013 at 10.3% and the lowest occurring in 2020 at 7.6%. We can also see that the year with the highest percentage of CO2 emissions was not in the last year of the period (2020) but rather in 2013

# ## Total CO2 emissions for UK (Bar Chart Visualisation)


# Filter the dataframe to only include UK's data
uk_df = df[df['iso_code'] == 'GBR']

# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
uk_df = uk_df[cols]

# Remove rows with missing data
uk_df = uk_df.dropna()

# Convert the 'year' column to datetime format and set it as the index
uk_df['year'] = pd.to_datetime(uk_df['year'], format='%Y')
uk_df = uk_df.set_index('year')

# Select the last 10 years of data
uk_last_10 = uk_df.loc['2011':'2020']

# Create a bar chart
plt.bar(uk_last_10.index.year, uk_last_10['co2_per_capita'], color= colors)
plt.title('CO2 Emissions per Capita in the UK (Last 10 Years)', fontsize = 12)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize = 12)
plt.show()


# ### From the bar chart, we can see the trend of CO2 emissions per capita in the UK over the last 10 years. We can observe that there is a slight decrease in CO2 emissions per capita from 2011 to 2016, followed by an increase until 2019, and then a decrease in 2020. This may suggest that efforts to reduce carbon emissions in the UK have had some success, but there is still room for improvement. Additionally, the COVID-19 pandemic may have played a role in the decrease in CO2 emissions in 2020.

# ### Summary Statistics for the data


# Select the 'co2_per_capita' column and drop missing values
co2_data = df['co2_per_capita'].dropna()

# Calculate summary statistics
mean = co2_data.mean()
median = co2_data.median()
min_value = co2_data.min()
max_value = co2_data.max()
std = co2_data.std()
variance = co2_data.var()

# Print summary statistics
print('Mean:', mean)
print('Median:', median)
print('Min:', min_value)
print('Max:', max_value)
print('Standard Deviation:', std)
print('Variance:', variance)

# Visualize distribution of data using a histogram
plt.hist(co2_data, bins=50)
plt.title('Distribution of CO2 Emissions per Capita', fontsize = 12)
plt.xlabel('CO2 Emissions (metric tons per capita)', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.show()


# #### From the summary statistics, we can see that the mean CO2 emissions per capita is approximately 4.5 metric tons, while the median is around 1.5 metric tons. The minimum value is close to 0 metric tons, while the maximum value is over 55 metric tons. The standard deviation of the data is quite large, indicating that there is a wide range of variability in the CO2 emissions per capita across countries.
# 
# #### From the histogram, we can see that the distribution of CO2 emissions per capita is skewed to the right, with a long tail towards higher emissions. This indicates that there are some countries with much higher emissions than the majority of countries.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset locally on the computer
df = pd.read_csv('co2emission.csv')

# Select the required countries
countries = ['UK', 'USA', 'Russia', 'Qatar', 'China']
filtered_df = df[df['country'].isin(countries)]

# Filter the dataset to only include the last 10 years of data
last_10_years = filtered_df['year'] > filtered_df['year'].max() - 10
filtered_df = filtered_df[last_10_years]

# Create a line plot to compare CO2 emissions over time
sns.set_style('darkgrid')
plt.figure(figsize=(12, 6))
for country in countries:
    country_df = filtered_df[filtered_df['country'] == country]
    plt.plot(country_df['year'], country_df['co2'], label=country)
plt.title('CO2 Emissions of Selected Countries (Last 10 Years)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
plt.legend()
plt.show()


# ### This line plot above compares the CO2 emissions over time for selected countries. The plot shows the trend of CO2 emissions over the last 10 years for each country, with a different line representing each country. The plot provides insights into the differences in CO2 emissions between countries and how they have changed over time. It also highlights which countries have had the highest and lowest CO2 emissions over the last decade.


# Create a scatter plot to compare energy consumption and CO2 emissions
plt.figure(figsize=(12, 6))
for country in countries:
    country_df = filtered_df[filtered_df['country'] == country]
    plt.scatter(country_df['energy_per_capita'], country_df['co2'], label=country)
plt.title('Energy Consumption vs CO2 Emissions of Selected Countries', fontsize=14)
plt.xlabel('Energy Consumption (kg of oil equivalent per capita)', fontsize=12)
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
plt.legend()
plt.show()


# ### This scatter plot above compares energy consumption and CO2 emissions for the selected countries. From the plot, it can be seen that there is a positive correlation between energy consumption and CO2 emissions. Countries with higher energy consumption tend to have higher CO2 emissions as well. The scatter plot shows that Qatar has the highest energy consumption and CO2 emissions among the selected countries, followed by the United States, China, and Russia. The European countries appear to have lower energy consumption and CO2 emissions compared to the other regions. This plot can help us understand the relationship between energy consumption and CO2 emissions and identify the countries that contribute the most to global CO2 emissions.


# Create a scatter plot to compare GDP and CO2 emissions
plt.figure(figsize=(12, 6))
for country in countries:
    country_df = filtered_df[filtered_df['country'] == country]
    plt.scatter(country_df['gdp'], country_df['co2'], label=country)
plt.title('GDP vs CO2 Emissions of Selected Countries', fontsize=14)
plt.xlabel('GDP (current US$)', fontsize=12)
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
plt.legend()
plt.show()


# ### This scatter plot compares the GDP and CO2 emissions for the selected countries. The plot suggests a positive correlation between the two variables, meaning that as a country's GDP increases, so does its CO2 emissions. This is not surprising as industrialization and economic growth often require increased energy consumption, which in turn leads to higher CO2 emissions.
# 
# ### However, there are some interesting outliers in the plot, such as China and India, which have relatively low GDP per capita but high CO2 emissions. This could be attributed to their large populations and heavy reliance on coal as a primary energy source. On the other hand, countries such as Qatar and the United Arab Emirates have high GDP per capita and high CO2 emissions, likely due to their large oil and gas reserves and heavy reliance on these resources for their economy.
# 
# ### Overall, the plot highlights the complex relationship between economic development and environmental sustainability, and the need for effective policy solutions to address climate change.


# Create a scatter plot to compare flaring and CO2 emissions
plt.figure(figsize=(12, 6))
for country in countries:
    country_df = filtered_df[filtered_df['country'] == country]
    plt.scatter(country_df['flaring_co2'], country_df['co2'], label=country)
plt.title('Flaring vs CO2 Emissions of Selected Countries', fontsize=14)
plt.xlabel('Flaring (thousand cubic meters)', fontsize=12)
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
plt.legend()
plt.show()


# ### This scatter plot compares the flaring and CO2 emissions of selected countries. Flaring is the process of burning off unwanted natural gas during oil extraction, and it is a significant source of greenhouse gas emissions. The x-axis represents flaring in thousand cubic meters, and the y-axis represents CO2 emissions in metric tons per capita.
# 
# ### From the visualization, we can see that most countries have low flaring and low CO2 emissions, indicating that they may not have significant oil and gas extraction activities. However, some countries, such as Russia, Iraq, and Kuwait, have high flaring and high CO2 emissions, indicating that they have significant oil and gas extraction activities.
# 
# ### Additionally, we can see that there is a positive correlation between flaring and CO2 emissions, as countries with higher flaring tend to have higher CO2 emissions. This is expected since flaring releases greenhouse gases, contributing to overall emissions.
# 
# ### The visualization highlights the importance of reducing flaring during oil and gas extraction to reduce greenhouse gas emissions and combat climate change.

# ## Visualisation Dashboard


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Read in data
df = pd.read_csv('co2emission.csv')

# Define color palette
colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']

# Set up the grid
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Dashboard Visualization of CO2 Emissions - Udoye Cynthia Chinenye - 22029346', fontweight='bold',fontsize=16, y=2.3, x= 0.40)
fig.patch.set_facecolor('lightgray')
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

# Plot 1: CO2 emissions by country in 2017
ax1 = fig.add_subplot(gs[0, 0])
df_2017 = df[df['year']==2017]
df_sorted = df_2017.sort_values(by='co2', ascending=False).head(10)
ax1.bar(df_sorted['country'], df_sorted['co2'], color=colors)
ax1.set_title('Top 10 Countries with the Highest CO2 Emissions in 2017', fontsize=12)
ax1.set_xlabel('Country')
ax1.set_ylabel('CO2 Emissions (metric tons per capita)')
ax1.tick_params(labelrotation=90)
ax1.set_facecolor('white')

# Plot 2: CO2 emissions trend for top 5 countries from 1990 to 2017
ax2 = fig.add_subplot(gs[0, 1])
df_top5 = df[df['country'].isin(df_sorted['country'].unique())]
df_top5 = df_top5.pivot(index='year', columns='country', values='co2')
df_top5.plot(ax=ax2, color=colors, lw=2.5)
ax2.set_title('CO2 Emissions Trend for Top 5 Countries from 1990 to 2017', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('CO2 Emissions (metric tons per capita)')
ax2.set_facecolor('white')

# Plot 3: CO2 emissions scatter plot for top 5 countries in 2017
ax3 = fig.add_subplot(gs[1, 0])
df_sorted = df_2017.sort_values(by='co2', ascending=False).head(5)
ax3.scatter(df_sorted['country'], df_sorted['co2'], color='steelblue', alpha=0.8)
ax3.set_title('CO2 Emissions by Top 5 Countries in 2017', fontsize=12)
ax3.set_xlabel('Country')
ax3.set_ylabel('CO2 Emissions (metric tons per capita)')
ax3.tick_params(labelrotation=90)
ax3.set_facecolor('white')


# Plot 4: CO2 emissions trend for selected countries from 1990 to 2017
ax4 = fig.add_subplot(gs[1, 1])
selected_countries = ['United States', 'China', 'India', 'Russia', 'Germany']
df_selected = df[df['country'].isin(selected_countries)]
df_selected = df_selected.pivot(index='year', columns='country', values='co2')
df_selected.plot(ax=ax4, color=colors, lw=2.5)
ax4.set_title('CO2 Emissions Trend for Selected Countries from 1990 to 2017', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('CO2 Emissions (metric tons per capita)')
ax4.set_facecolor('white')

# Adjust spacing between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=2.2, wspace=0.3, hspace=0.5)

# Save and show the plot
plt.savefig('co2_emissions_dashboard.png', dpi=300, facecolor=fig.get_facecolor())
#plt.show()


# ## Here are some insights that can be gained from the visualization above:
# 
# Plot 1 shows the top 10 countries with the highest CO2 emissions in 2017. China had the highest CO2 emissions by a significant margin, followed by the United States and India.
# 
# Plot 2 displays the CO2 emissions trend for the top 5 countries from 1990 to 2017. All countries showed an increasing trend in CO2 emissions, with China's emissions increasing at a much faster rate than the other countries.
# 
# Plot 3 shows a scatter plot of CO2 emissions by country in 2017. There is a wide range of emissions across countries, with some countries having very low emissions and others having very high emissions.
# 
# Plot 4 displays the CO2 emissions trend for selected countries from 1990 to 2017. The selected countries are the top 3 emitters (China, United States, and India) and two other large emitters (Russia and Germany). The plot shows that China's emissions have increased significantly over time, while the other countries have had a more moderate increase.
# 
# The plots indicate that there is a correlation between a country's population and its CO2 emissions. Countries with larger populations tend to have higher CO2 emissions.
# 
# The plots suggest that there is a need for more concerted efforts to reduce global CO2 emissions, given the increasing trend in emissions across countries over time.

# ## Interactive Visualisation Dashboard - This gives room for interaction into the dataset


plt.style.use('fivethirtyeight')


#Calling df.interactive() on a dataframe can convert it into an interactive visualization object that can be 
#further customized and manipulated using other methods and tools provided by the library.
intr = df.interactive()


#Create a Panel widget to navigate and display data for different years. 
default_slider = pn.widgets.IntSlider(start=1750, end=2020, step=4, value=2020, bar_color='#800000', name='Year')
default_slider


#This line creates a radio button group using Panel library, with the name
#"co2_co2" and options "gdp" and "co2_per_capita". The button group has a green "success" theme. 
co2_co2 = pn.widgets.RadioButtonGroup(name = 'Y-axis',
                                        options = ['co2','co2_per_capita'], #values 
                                        button_type = 'success') # designtheme


#connecting datapipeline and the widgets for places and continents to be considered 
continents = ['World', 'Asia', 'Oceania', 'Europe', 'Africa', 'North America', 'South America', 'Antarctica']

# Filter and aggregate the data to get the average CO2 emissions by year and country for selected continents
design_path = (
    intr[ (intr.year <= default_slider) & 
         (intr.country.isin(continents))
       ]
    .groupby(['year','country'])[co2_co2].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='year')
    .reset_index(drop=True)
)


#Display 
design_path


# import modules
import holoviews as hv
from holoviews import opts

# This line enables the Bokeh plotting backend for HoloViews
hv.extension('bokeh')

# Create a Holoviews plot to visualize CO2 emissions by country and year
# Co2 by geographical region 
co2_by_geo = design_path.hvplot(x='year', y=co2_co2, by='country', line_width=2, title="Geographical Regions CO2 emission")
co2_by_geo.opts(opts.Curve(bgcolor='lightgray', show_grid=True, gridstyle={'grid_line_color': 'white'}))


# ## CO2 emission over a particular period of time.


# Create a Tabulator widget to display CO2 emissions over time for each country in the filtered dataset
# The widget has remote pagination enabled and displays 8 rows per page
co2_overtime = design_path.pipe(pn.widgets.Tabulator, pagination='remote', page_size=8, sizing_mode='stretch_width', background='#f2f2f2')


#Display the Widget
co2_overtime


# ## Energy Per Capital vs CO2


# This code creates a dataframe of CO2 emissions for countries outside of the selected continents,
#for the default_slider value, grouped by country and energy consumption per capita
energyco2pipeline_for_scatter = (
    intr[
        ( intr.year == default_slider) &
        (~ (intr.country.isin(continents)))
    ]
    .groupby(['country', 'year', 'energy_per_capita'])['co2'].mean()
    .to_frame()
    .reset_index()
    .sort_values(by='year')  
    .reset_index(drop=True)
)


#Display the pipeline
energyco2pipeline_for_scatter


# Create a scatter plot of CO2 emissions versus energy per capita by country
co2gdp_scatter = energyco2pipeline_for_scatter.hvplot(x='energy_per_capita', 
                                                                y='co2', 
                                                                by='country', 
                                                                size=80, kind="scatter", 
                                                                alpha=0.7,
                                                                legend=False, 
                                                                height=600, 
                                                                width=500)


# Display the scatter plot
co2gdp_scatter


# ## CO2 by Continents in the Dataset (These are sources for CO2 emission)


# Create a radio button group for selecting CO2 sources
co2_sources = pn.widgets.RadioButtonGroup(
    name='Y axis', 
    options=['coal_co2', 'oil_co2', 'gas_co2'], 
    button_type='success'
)

# Define a list of continents
continents = ['Asia', 'Oceania', 'Europe', 'Africa', 'North America', 'South America', 'Antarctica']

# Filter the data to include only the selected year and continents
pipelineforsources = (
    intr[
        (intr.year == default_slider) &
        (intr.country.isin(continents))
    ]
    # Group the data by year and country, and sum the CO2 emissions for each source
    .groupby(['year', 'country'])[co2_sources].sum()
    .to_frame()
    .reset_index()
    .sort_values(by='year')  
    .reset_index(drop=True)
)


#Display the c02 sources
co2_sources


# create a bar plot showing the elements generating CO2
barplot_forsources = pipelineforsources.hvplot(kind='bar', 
                                                     x='country', 
                                                     y=co2_sources, 
                                                     title='Elements generating CO2/SOurces',
                                                     color = '#A04000')
#Display the co2 sources
barplot_forsources


# This code is using a predefined layout template to organize the visualizations.
template = pn.template.FastListTemplate(
    title='World CO2 Emission Dashboard - Udoye Cynthia Chinenye- 22029346', 
    sidebar=[pn.pane.Markdown("# Carbon Dioxide Emission"), 
             pn.pane.Markdown("#### Carbon dioxide is a greenhouse gas, which means that it traps heat in the Earth's atmosphere and contributes to global warming and climate change. As the concentration of carbon dioxide in the atmosphere increases, it contributes to rising global temperatures, which can have far-reaching impacts on the planet and its inhabitants, including rising sea levels, more extreme weather events, and changes in ecosystems and agriculture."), 
             pn.pane.PNG('earth.png', sizing_mode='scale_both'),
             pn.pane.Markdown("## 22029346"),   
             default_slider],  # adds a slider to select the year
    main=[pn.Row(pn.Column(co2_co2, 
                          co2_by_geo.panel(width=500), margin=(0,25)), 
                 co2_overtime.panel(width=500)),  # adds a plot of CO2 emissions by country and a table of CO2 emissions over time
          pn.Row(pn.Column(co2gdp_scatter.panel(width=500), margin=(0,25)), 
                 pn.Column(co2_sources, barplot_forsources.panel(width=500)))],  # adds a scatter plot of CO2 emissions vs energy consumption and a bar plot of CO2 emissions by energy source
    accent_base_color="#FF5733",
    header_background="#273746"
)

# sets the background color of the sidebar to light blue
template.sidebar.background = '#ADD8E6'

# displays the dashboard as an interactive web page
template.show() 

# displays the dashboard as a servable application
#template.servable();  


# ### Insights gleaned from the data (Textbox)

# 1.Coal is the largest contributor to CO2 emissions among the energy sources analyzed.
# 
# 2.Oil is the second-largest contributor, but its emissions are still significantly lower than those of coal.
# 
# 3.Gas, cement, and flaring also contribute to CO2 emissions, but to a much lesser extent than coal and oil.
# Nuclear and hydroelectric power generate much lower levels of CO2 emissions compared to the other sources analyzed.
# 
# 4.There seems to be a positive correlation between energy consumption per capita and CO2 emissions. Countries with higher energy consumption per capita tend to have higher CO2 emissions.
# 
# 5.There are a few outliers in the plot, with some countries having very high CO2 emissions compared to their energy consumption per capita. These countries could potentially be using inefficient energy sources or have a high dependence on carbon-intensive industries.
# 
# 6.The majority of the data points cluster towards the lower end of the x-axis, indicating that many countries have relatively low energy consumption per capita. However, even with this low consumption, some of these countries still have relatively high CO2 emissions, which could indicate a reliance on carbon-intensive industries or a lack of clean energy sources.
# 
# 6.Some countries have a steady increase in CO2 emissions over time, while others have a more fluctuating pattern.
# 
# 7.Some countries have higher CO2 emissions than others, which could be due to differences in industrialization, population size, or natural resources.

# ### Insights gleaned with visualisations.


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CO2 dataset
df = pd.read_csv('co2emission.csv')

# 1. Coal is the largest contributor to CO2 emissions among the energy sources analyzed.
plt.figure(figsize=(12, 14))
ax = sns.barplot(data=df, x='coal_co2', y='country', color='gray')
ax.set_title('CO2 Emissions from Coal', fontsize=14)
ax.set_xlabel('CO2 Emissions (metric tons per capita)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.tick_params(axis='y', labelsize=6)  # adjust font size of y-axis tick labels
plt.show()


# ### This visualization shows the CO2 emissions from coal for each country in the dataset, with the countries sorted by decreasing emissions. The height of each bar represents the total CO2 emissions from coal per capita for that country. The bar for China is particularly striking, indicating that its CO2 emissions from coal are far higher than any other country in the dataset. This suggests that reducing coal consumption and finding alternative sources of energy would be an effective strategy for reducing global CO2 emissions.


# 2. Oil is the second-largest contributor, but its emissions are still significantly lower than those of coal.
plt.figure(figsize=(12, 14))
ax = sns.barplot(data=df, x='oil_co2', y='country', color='gray')
ax.set_title('CO2 Emissions from Oil', fontsize=14)
ax.set_xlabel('CO2 Emissions (metric tons per capita)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.tick_params(axis='y', labelsize=6)  # adjust font size of y-axis tick labels
plt.show()


# ### This visualization shows the CO2 emissions from oil for each country in the dataset, with the height of each bar representing the amount of CO2 emissions per capita. The insight here is that oil is the second-largest contributor to CO2 emissions, following coal. However, the emissions from oil are still significantly lower than those from coal. This suggests that reducing the use of coal and transitioning to cleaner energy sources could have a larger impact on reducing CO2 emissions compared to reducing the use of oil.


# 3. Gas, cement, and flaring also contribute to CO2 emissions, but to a much lesser extent than coal and oil. 
plt.figure(figsize=(14, 12))
ax = sns.barplot(data=df, x='gas_co2', y='country', color='gray')
ax.set_title('CO2 Emissions from Gas', fontsize=14)
ax.set_xlabel('CO2 Emissions (metric tons per capita)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.tick_params(axis='y', labelsize=6)  # adjust font size of y-axis tick labels
plt.show()

plt.figure(figsize=(14, 12))
ax = sns.barplot(data=df, x='cement_co2', y='country', color='gray')
ax.set_title('CO2 Emissions from Cement Production', fontsize=14)
ax.set_xlabel('CO2 Emissions (metric tons per capita)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.tick_params(axis='y', labelsize=6)  # adjust font size of y-axis tick labels
plt.show()

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df, x='flaring_co2', y='country', color='gray')
ax.set_title('CO2 Emissions from Gas Flaring', fontsize=14)
ax.set_xlabel('CO2 Emissions (metric tons per capita)', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.tick_params(axis='y', labelsize=6)  # adjust font size of y-axis tick labels
plt.show()


# ### The insight from these visualizations is that gas, cement, and flaring are significant sources of CO2 emissions for some countries, but contribute much less compared to coal and oil. The bar charts show the total CO2 emissions from each source for each country, allowing us to compare the relative contributions of each source across countries. These visualizations suggest that while these sources are important, they are not the primary drivers of CO2 emissions for most countries in this dataset.


# 4. There seems to be a positive correlation between energy consumption per capita and CO2 emissions.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='energy_per_capita', y='co2_per_capita', color='gray')
plt.title('CO2 Emissions vs Energy Consumption per Capita')
plt.xlabel('Energy Consumption per Capita (kWh)')
plt.ylabel('CO2 Emissions per Capita (metric tons)')
plt.show()


# ### The insight from this visualization is that there appears to be a positive correlation between energy consumption per capita and CO2 emissions. As energy consumption per capita increases, so do CO2 emissions per capita. This suggests that countries with higher energy consumption tend to have higher CO2 emissions. However, the relationship is not strictly linear, as there are some outliers in the plot where countries have higher CO2 emissions compared to their energy consumption per capita.


# 5. There are a few outliers in the plot, with some countries having very high CO2 emissions compared to their energy consumption per capita.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='energy_per_capita', y='co2_per_capita', color='gray')
plt.title('CO2 Emissions vs Energy Consumption per Capita')
plt.xlabel('Energy Consumption per Capita (kWh)')
plt.ylabel('CO2 Emissions per Capita (metric tons)')
plt.xlim((0, 20000))
plt.ylim((0, 100))
plt.show()


# ### The insight from this visualization is that there are certain countries that have a relatively high level of CO2 emissions compared to their energy consumption per capita, indicating that they may have less efficient or more carbon-intensive energy systems. These countries are represented as outliers in the scatterplot, which is useful for identifying countries that may need to make improvements in their energy systems to reduce their carbon footprint. The x and y-axis limits have been set to better visualize the majority of the data points, but these limits can be adjusted as needed to focus on specific areas of the plot.


# 6. The majority of the data points cluster towards the lower end of the x-axis, indicating that many countries have relatively low energy consumption per capita. 
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='energy_per_capita', color='gray', bins=20)
plt.title('Distribution of Energy Consumption per Capita')
plt.xlabel('Energy Consumption per Capita (kWh)')
plt.ylabel('Frequency')
plt.show()


# ### The insight gleaned from this visualization is that a large proportion of the countries in the dataset have relatively low energy consumption per capita, as the majority of the data points are clustered towards the lower end of the x-axis. This could indicate that many countries have limited access to energy or are not consuming energy at the same rate as more developed countries.


# 7. Some countries have a steady increase in CO2 emissions over time, while others have a more fluctuating pattern.
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='year', y='co2', hue='country', linewidth=1)
plt.title('CO2 Emissions Over Time by Country', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 Emissions (million metric tons)', fontsize=12)
plt.tick_params(axis='both', labelsize=10) # adjust font size of both axes' tick labels
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # add legend outside plot area
plt.show()


# ### This visualization shows how CO2 emissions have changed over time for each country in the dataset. The insight gleaned is that some countries, such as China and the United States, have had a relatively steady increase in CO2 emissions over time, while others, such as India and Russia, have had a more fluctuating pattern. This information can be useful in identifying trends and patterns in emissions over time and can inform policy decisions regarding reducing emissions.




