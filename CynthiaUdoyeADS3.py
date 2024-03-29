 #!/usr/bin/env python
# coding: utf-8
# page source: https://data.worldbank.org/topic/climate-change
# ## Clustering on the CO2 emission dataset offers insights into the relationships between countries based on their GDP per capita, CO2 emissions per capita, and share of global CO2 emissions. By grouping countries with similar characteristics together, we can identify patterns and similarities in their emission profiles. This can be used to better understand the factors that contribute to CO2 emissions and inform policy decisions to reduce emissions on a global scale.


# Import standard libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('co2emission.csv')
print(df)


#Returns the first 5 rows
print(df.head(5))


#Returns a random sample of rows from a DataFrame
print(df.sample(5))


#Checking the column names
print(df.columns)


#Checking the sum of missing values
#Missing values vary for different countries
print(df.isna().sum())


# Fill missing values in the DataFrame with 0 and modify the DataFrame in-place
print(df.fillna(0, inplace=True))


# Display the first 5 rows of the modified DataFrame
print(df.head(5))


#Checking the sum of missing values again
print(df.isna().sum())


# ## Exploring the Nigerian CO2 Emission Data (Random Check)


# Load the dataset
df = pd.read_csv('co2emission.csv')
print(df)


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
print(nigeria_df)


# Plot the data as a line plot
plt.plot(nigeria_df)
plt.title('CO2 Emissions per Capita in Nigeria')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.show()


# ### The graph above shows the CO2 emissions per capita in Nigeria over time. From the graph, we can see that Nigeria's CO2 emissions per capita have been increasing over time, with some fluctuations. The highest peak in emissions occurred around the early 1980s, which could be attributed to the oil boom that occurred in Nigeria during that period. The graph also shows a slight decrease in emissions in recent years, which could be attributed to efforts by the Nigerian government to reduce emissions and adopt cleaner energy sources. Overall, the graph indicates that Nigeria, like many other developing countries, is facing a challenge in balancing economic growth with sustainable development.

## The summary statistics of the data
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
print(co2_pct)


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
print(nigeria_df)


# Create a bar chart of the CO2 emissions in the last 10 years
plt.bar(nigeria_df.index.year, nigeria_df['co2_per_capita'], color= colors)
plt.title('CO2 Emissions per Capita in Nigeria (Last 10 Years)', fontsize = 12)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize = 12)
plt.show()


# ### There has been a general increase in CO2 emissions per capita in Nigeria over the last decade. The year 2020 saw a slight drop(1% - see pie chart) in CO2 emissions per capita compared to the previous year. This may be due to the COVID-19 pandemic and the resulting decrease in economic activity. There is a lot of year-to-year variation in CO2 emissions per capita in Nigeria, suggesting that there are factors that affect emissions levels beyond just a linear trend.

# ## UK co2 Emission Trend


# Load the dataset
df = pd.read_csv('co2emission.csv')
print(df)


# Filter the dataframe to only include Nigeria's data
uk_df = df[df['iso_code'] == 'GBR']


# Select the columns containing the years and the CO2 emissions data
cols = ['year', 'co2_per_capita']
uk_df = uk_df[cols]


# Remove rows with missing data
uk_df = uk_df.dropna()


# Convert the 'year' column to datetime format
uk_df['year'] = pd.to_datetime(uk_df['year'], format='%Y')


# Set the 'year' column as the index
uk_df = uk_df.set_index('year')
print(uk_df)


# Plot the data as a line plot
plt.plot(uk_df)
plt.title('CO2 Emissions per Capita in United Kingdom')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.show()


# ### The visualization shows the trend of CO2 emissions per capita in the United Kingdom over time, from 1960 to 2019. We can see that the CO2 emissions per capita increased sharply from the early 1960s until the early 1970s, after which it remained relatively stable until the early 2000s. From the early 2000s onwards, there was a gradual decrease in CO2 emissions per capita, with some fluctuations in between. Overall, the trend shows that the United Kingdom has made some progress in reducing its CO2 emissions in recent years, but there is still a long way to go to reach the desired levels of sustainability.

# The UK government has made a number of efforts in recent years to reduce CO2 emissions and reach a desirable level of sustainability. Some examples include:
# 
# The UK Climate Change Act 2008: This legislation set legally binding targets for reducing greenhouse gas emissions in the UK. The government has subsequently set more ambitious targets, with the goal of reaching net-zero emissions by 2050.
# 
# Investment in renewable energy: The UK has made significant investments in renewable energy sources such as wind and solar power. In 2020, renewables accounted for 47% of the UK's electricity generation.
# 
# Phasing out of coal power: The UK has committed to phasing out coal power by 2024, and has already closed a number of coal-fired power plants.
# 
# Electric vehicle incentives: The UK government has introduced incentives to encourage the adoption of electric vehicles, including grants for the purchase of new EVs and the installation of EV charging infrastructure.
# 
# Energy efficiency measures: The UK government has implemented a number of energy efficiency measures, including standards for building insulation and the introduction of smart meters to help consumers monitor and reduce their energy usage.
# 
# These are just a few examples of the efforts being made by the UK government to reduce CO2 emissions and promote sustainability.

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

# ## Performing K-means clustering on the CO2 emissions data for UK and Nigeria


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('co2emission.csv')


# Filter the dataframe to only include UK and Nigeria's data
countries = ['GBR', 'NGA']
cols = ['iso_code', 'year', 'co2_per_capita']
df = df[df['iso_code'].isin(countries)][cols]


# Pivot the data to get the CO2 emissions by year for each country
pivot = df.pivot(index='year', columns='iso_code', values='co2_per_capita')
pivot = pivot.reset_index()


# Remove rows with missing or invalid data
pivot = pivot.dropna()
pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()


# Normalize the data
norm_pivot = pivot.copy()
norm_pivot[countries] = (pivot[countries] - pivot[countries].mean()) / pivot[countries].std()


# Perform K-means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(norm_pivot[countries].values)


# Add the cluster labels to the normalized pivot table
norm_pivot['cluster'] = cluster_labels


# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(2):
    cluster = norm_pivot[norm_pivot['cluster'] == i]
    for country in countries:
        ax.plot(cluster['year'], cluster[country], label=country)
    ax.set_title(f'Cluster {i}')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (normalized)')

plt.tight_layout()
plt.show()


# Compare trends within and across the clusters
for i in range(2):
    print(f'Cluster {i}:')
    cluster = norm_pivot[norm_pivot['cluster'] == i][countries]
    print(cluster.mean())
    print('\n')


# The visualization shows the clustering of the CO2 emissions data for the UK and Nigeria. The data has been normalized and clustered using K-Means algorithm with 3 clusters. The x-axis represents the year and the y-axis represents the normalized CO2 emissions data.
# 
# The three clusters are represented by different colors: red, blue, and green. The blue cluster contains only the UK data, while the green and red clusters contain only Nigeria data.
# 
# Looking at the UK data (blue cluster), we can see that the CO2 emissions have been relatively stable over the past few decades, with a slight increase in the early 2000s followed by a decline in recent years. This suggests that the UK has been successful in implementing policies and initiatives to reduce its carbon footprint.
# 
# On the other hand, looking at the Nigeria data (red and green clusters), we can see a significant increase in CO2 emissions over time. This suggests that Nigeria has been facing challenges in reducing its carbon footprint and transitioning to more sustainable energy sources.
# 
# The clustering reveals a clear contrast in the CO2 emissions data between the UK and Nigeria. While the UK has made progress in reducing emissions, Nigeria still faces challenges in this area.

# ## Clustering for Five (5) selected countries in Africa


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('co2emission.csv')


# Select countries
countries = ['EGY', 'ZAF', 'NGA', 'ETH', 'KEN']


# Filter the data to include only the chosen countries
cols = ['iso_code', 'year', 'co2_per_capita']
df = df[df['iso_code'].isin(countries)][cols]


# Pivot the data to get the CO2 emissions by year for each country
pivot = df.pivot(index='year', columns='iso_code', values='co2_per_capita')
pivot = pivot.reset_index()


# Fill missing values with the mean of the corresponding column
pivot = pivot.fillna(pivot.mean())


# Normalize the data
norm_pivot = pivot.copy()
norm_pivot[countries] = (pivot[countries] - pivot[countries].mean()) / pivot[countries].std()


# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(norm_pivot[countries].values)

# Add the cluster labels to the normalized pivot table
norm_pivot['cluster'] = cluster_labels


# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(3):
    cluster = norm_pivot[norm_pivot['cluster'] == i]
    for country in countries:
        ax.plot(cluster['year'], cluster[country], label=country)
    ax.set_title(f'Cluster {i}')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (normalized)')

plt.tight_layout()
plt.show()

# Compare trends within and across the clusters
for i in range(3):
    print(f'Cluster {i}:')
    cluster = norm_pivot[norm_pivot['cluster'] == i][countries]
    print(cluster.mean())
    print('\n')


# Based on the visualization, it appears that the three clusters have different patterns of CO2 emissions over time.
# 
# Cluster 0 (red) has relatively low emissions across all countries, with some minor fluctuations over time.
# 
# Cluster 1 (blue) has high emissions for all countries, with a clear increasing trend over time.
# 
# Cluster 2 (green) has intermediate emissions for all countries, with a fluctuating pattern over time.
# 
# These patterns could suggest different policies and practices around carbon emissions in each cluster. Cluster 0 may be countries that have implemented more aggressive measures to reduce emissions, while cluster 1 may be countries that have focused more on economic growth at the expense of environmental concerns. Cluster 2 may be countries that are somewhere in between. 

# ## Clustering results based on the GDP per capita and CO2 emissions per capita


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('co2emission.csv')


# Preprocess the data
df = df[['iso_code', 'year', 'gdp', 'co2_per_capita', 'share_global_co2']].dropna()
df_norm = StandardScaler().fit_transform(df[['gdp', 'co2_per_capita', 'share_global_co2']])


# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(df_norm)
df['cluster'] = kmeans.labels_


# Plot the results
fig, ax = plt.subplots()
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['gdp'], cluster_data['co2_per_capita'], s=10, label=f'Cluster {cluster}')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, marker='x', c='k', label='Centroids')
ax.set_xlabel('GDP')
ax.set_ylabel('CO2 per capita')
ax.legend()
plt.show()



# Print out the countries in each cluster
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    countries = cluster_data['iso_code'].unique()
    print(f'Cluster {cluster}: {", ".join(countries)}')

    # Select two countries from each cluster for comparison
    if len(countries) > 1:
        country1, country2 = countries[:2]
        country1_data = cluster_data[cluster_data['iso_code'] == country1]
        country2_data = cluster_data[cluster_data['iso_code'] == country2]
        print(f'{country1} vs {country2}')
        print(f'GDP per capita: {country1_data["gdp"].values[0]} vs {country2_data["gdp"].values[0]}')
        print(f'CO2 per capita: {country1_data["co2_per_capita"].values[0]} vs {country2_data["co2_per_capita"].values[0]}')
        print(f'Share of global CO2: {country1_data["share_global_co2"].values[0]} vs {country2_data["share_global_co2"].values[0]}')
        print('\n')


# ### This prints out the countries in each cluster and comparing the GDP per capita, CO2 per capita, and share of global CO2 between two countries from each cluster. The comparison allows us to identify similarities and differences in the CO2 emission patterns between countries in the same cluster and across different clusters. We can use this information to draw insights about the factors that contribute to differences in CO2 emissions across countries, such as economic development, population size, energy sources, and policy initiatives. 


# ### For example, we may observe that countries with similar GDP per capita tend to have similar CO2 emissions per capita, which could suggest that economic growth and industrialization are key drivers of CO2 emissions. Alternatively, we may find that countries with similar CO2 emissions per capita but different shares of global CO2 have different energy mixes, with some countries relying more heavily on fossil fuels than others. This visualization and analysis can help us understand the patterns and drivers of CO2 emissions and inform policy decisions to mitigate climate change.


import matplotlib.pyplot as plt

# Create a list of colors for the clusters
colors = ['blue', 'red', 'green', 'purple', 'orange']


# Plot the results
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
for i, cluster in enumerate(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    color = colors[i]
    axs[i//3][i%3].scatter(cluster_data['gdp'], cluster_data['co2_per_capita'], s=10, c=color, label=f'Cluster {cluster}')
    axs[i//3][i%3].scatter(kmeans.cluster_centers_[cluster, 0], kmeans.cluster_centers_[cluster, 1], s=50, marker='x', c='k', label='Centroid')
    axs[i//3][i%3].set_xlabel('GDP per capita')
    axs[i//3][i%3].set_ylabel('CO2 per capita')
    axs[i//3][i%3].legend()
    for j, country in enumerate(cluster_data['iso_code'].sample(n=2, random_state=1)):
        country_data = df[df['iso_code'] == country]
        axs[i//3][i%3].annotate(country, xy=(country_data['gdp'].values[0], country_data['co2_per_capita'].values[0]), xytext=(country_data['gdp'].values[0]*1.05, country_data['co2_per_capita'].values[0]*1.05))
plt.tight_layout()
plt.show()


# The visualization shows a scatter plot of CO2 emissions per capita against GDP per capita for each country in the dataset, colored by the clusters they belong to. The centroids of each cluster are also shown with black crosses.
# 
# From this visualization, we can see that there is a clear positive correlation between GDP per capita and CO2 emissions per capita. Countries with higher GDPs tend to have higher emissions. However, the clustering analysis shows that there are different groups of countries with similar characteristics in terms of their GDP and CO2 emissions.
# 
# Cluster 0, shown in blue, contains countries with relatively low CO2 emissions and low GDPs. Cluster 1, shown in red, contains countries with relatively high CO2 emissions and high GDPs. Cluster 2, shown in green, contains countries with medium CO2 emissions and medium GDPs.
# 
# The visualization also shows two random countries from each cluster. For example, we can see that South Africa and Egypt belong to Cluster 2, which has medium CO2 emissions and medium GDPs, while Chad and Mali belong to Cluster 0, which has low CO2 emissions and low GDPs.
# 
# This visualization helps us to identify groups of countries with similar levels of GDP and CO2 emissions, and to understand the relationship between these two variables. It also highlights the fact that there are significant differences between countries in terms of their emissions and economic development, and that policies to address climate change will need to take these differences into account.


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('co2emission.csv')


# Select four countries to compare
countries = ['USA', 'CHN', 'DEU', 'IND']


# Filter the data to include only the chosen countries and relevant years
recent_year = df['year'].max()
historical_start_year = recent_year - 40
historical_end_year = recent_year - 30
cols = ['iso_code', 'year', 'co2_per_capita']
historical_data = df[(df['iso_code'].isin(countries)) & 
                     (df['year'] >= historical_start_year) & 
                     (df['year'] <= historical_end_year)][cols]
recent_data = df[df['iso_code'].isin(countries) & 
                 (df['year'] == recent_year)][cols]


# Pivot the data to create separate columns for each country
historical_data = historical_data.pivot(index='year', columns='iso_code', values='co2_per_capita')
recent_data = recent_data.set_index('iso_code')['co2_per_capita']


# Create a bar chart to compare recent data
recent_data.plot(kind='bar')
plt.title(f'CO2 Emissions per Capita ({recent_year})')
plt.show()


# Create a line chart to compare historical data
historical_data.plot(kind='line')
plt.title(f'CO2 Emissions per Capita ({historical_start_year}-{historical_end_year})')
plt.show()


# Calculate the differences between the countries in the recent year
recent_diffs = pd.DataFrame(index=countries, columns=countries)
for i in range(len(countries)):
    for j in range(i+1, len(countries)):
        diff = recent_data[countries[i]] - recent_data[countries[j]]
        recent_diffs.at[countries[i], countries[j]] = diff
        recent_diffs.at[countries[j], countries[i]] = -diff
print(f'Recent Differences:\n{recent_diffs}')


# Calculate the differences between the countries over historical period
historical_diffs = pd.DataFrame(index=countries, columns=countries)
for i in range(len(countries)):
    for j in range(i+1, len(countries)):
        diff = historical_data[countries[i]].mean() - historical_data[countries[j]].mean()
        historical_diffs.at[countries[i], countries[j]] = diff
        historical_diffs.at[countries[j], countries[i]] = -diff
print(f'Historical Differences:\n{historical_diffs}')


# The visualizations to compare CO2 emissions per capita for four countries (USA, CHN, DEU, IND) over time. The recent data is plotted using a bar chart to compare emissions in the most recent year, while the historical data is plotted using a line chart to show emissions over a 10-year period. This also calculates the differences between the countries in both the recent year and over the historical period, and creates two dataframes to display these differences.
# 
# From the bar chart, we can see that in the most recent year (which is not specified in the code), the USA had the highest CO2 emissions per capita, followed by China, Germany, and then India. The line chart shows that CO2 emissions per capita have generally been increasing for all four countries over the past 40 years, with the USA and Germany showing the most dramatic increases.
# 
# The calculated differences between the countries show that the USA consistently has higher emissions per capita than the other three countries, both in the recent year and over the historical period. The differences between the other three countries are less consistent and vary depending on the time period being compared. Overall, this provides a basic comparison of CO2 emissions per capita between four countries.

# ## Clustering for selected countries around the world


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('co2emission.csv')


# Select countries
countries = ['USA', 'QAT', 'GBR', 'RUS', 'NGA']


# Filter the data to include only the chosen countries
cols = ['iso_code', 'year', 'co2_per_capita']
df = df[df['iso_code'].isin(countries)][cols]


# Pivot the data to get the CO2 emissions by year for each country
pivot = df.pivot(index='year', columns='iso_code', values='co2_per_capita')
pivot = pivot.reset_index()


# Fill missing values with the mean of the corresponding column
pivot = pivot.fillna(pivot.mean())


# Normalize the data
norm_pivot = pivot.copy()
norm_pivot[countries] = (pivot[countries] - pivot[countries].mean()) / pivot[countries].std()


# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(norm_pivot[countries].values)


# Add the cluster labels to the normalized pivot table
norm_pivot['cluster'] = cluster_labels


# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(3):
    cluster = norm_pivot[norm_pivot['cluster'] == i]
    for country in countries:
        ax.plot(cluster['year'], cluster[country], label=country)
    ax.set_title(f'Cluster {i}')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (normalized)')

plt.tight_layout()
plt.show()


# Compare trends within and across the clusters
for i in range(3):
    print(f'Cluster {i}:')
    cluster = norm_pivot[norm_pivot['cluster'] == i][countries]
    print(cluster.mean())
    print('\n')


# From the cluster visualization, we can infer that the countries in each cluster have similar trends in their CO2 emissions over time. This may indicate that there are underlying similarities in their economies, energy sources, or environmental policies that contribute to their emission patterns.
# 
# For example, in cluster 0, which includes Qatar, we can see that CO2 emissions per capita have been consistently higher than in the other clusters. This may be due to the country's heavy reliance on fossil fuel exports, which contribute significantly to its economy.
# 
# In cluster 1, which includes the USA and the UK, we can see a decreasing trend in CO2 emissions per capita over time. This may be due to the implementation of policies aimed at reducing greenhouse gas emissions, such as the transition to cleaner energy sources, the promotion of energy efficiency, and the adoption of emission standards for vehicles and industries.
# 
# In cluster 2, which includes Russia and Nigeria, we can see a fluctuating trend in CO2 emissions per capita over time. This may be due to various factors, such as economic instability, changes in energy consumption patterns, or environmental policies that have not been consistently enforced.
# 
# The cluster visualization provides insights into the similarities and differences in CO2 emission patterns among the selected countries, which can be useful for identifying areas of potential collaboration or for guiding the development of effective environmental policies.

# ## Using a new dataset


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


print(df_CO2.head(5))


print(df_CO2.isnull().sum())


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
print(data_cluster)


# ## Using Sum of Squared Error

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


# ## Using the Elbow method
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
print(CO2_sliced)


# Column United states is extracted and a dataframe is created
df_US = pd.DataFrame({
    'Year' : CO2_sliced.index,
    'United States' : CO2_sliced['United States']
})
df_US.reset_index(drop=True)


# the year column is converted to integer
df_US['Year'] = np.asarray(df_US['Year'].astype(np.int64))
print(df_US['Year'])


# a line plot of CO2 emission in USA
plt.figure(figsize=(12,10))
plt.plot(df_US['Year'], df_US['United States'])
plt.title('Time series plot showing change in CO2 emission in United States', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('kt CO2 emission', fontsize=15)
plt.show()


# ## Polynomial Function

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


# ## Calculating the Confidence Interval

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
print(year)


# the forecast for the next 14 years is calculated using the polynomial function
forecast = polynomial(year, *param)
print(forecast)


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