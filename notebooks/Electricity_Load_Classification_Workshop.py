#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Import-data" data-toc-modified-id="Import-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import data</a></div><div class="lev2 toc-item"><a href="#If-you-have-got-the-data-stored-locally-on-your-computer" data-toc-modified-id="If-you-have-got-the-data-stored-locally-on-your-computer-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>If you have got the data stored locally on your computer</a></div><div class="lev2 toc-item"><a href="#If-working-with-colab" data-toc-modified-id="If-working-with-colab-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>If working with colab</a></div><div class="lev1 toc-item"><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis</a></div><div class="lev2 toc-item"><a href="#Check-data-index" data-toc-modified-id="Check-data-index-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Check data index</a></div><div class="lev2 toc-item"><a href="#Missing/null-values" data-toc-modified-id="Missing/null-values-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Missing/null values</a></div><div class="lev2 toc-item"><a href="#Visualise-some-sequences" data-toc-modified-id="Visualise-some-sequences-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Visualise some sequences</a></div><div class="lev2 toc-item"><a href="#Seasonality---do-we-have-any?" data-toc-modified-id="Seasonality---do-we-have-any?-24"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Seasonality - do we have any?</a></div><div class="lev2 toc-item"><a href="#Group-users-by-usage?" data-toc-modified-id="Group-users-by-usage?-25"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Group users by usage?</a></div><div class="lev2 toc-item"><a href="#Downsample-for-better-visualisation" data-toc-modified-id="Downsample-for-better-visualisation-26"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Downsample for better visualisation</a></div><div class="lev2 toc-item"><a href="#Other-things-that-can-be-done-during-preprocessing" data-toc-modified-id="Other-things-that-can-be-done-during-preprocessing-27"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Other things that can be done during preprocessing</a></div><div class="lev1 toc-item"><a href="#Preparation-for-Classification" data-toc-modified-id="Preparation-for-Classification-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Preparation for Classification</a></div><div class="lev2 toc-item"><a href="#Load-labels" data-toc-modified-id="Load-labels-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Load labels</a></div><div class="lev1 toc-item"><a href="#Random-forest" data-toc-modified-id="Random-forest-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Random forest</a></div><div class="lev2 toc-item"><a href="#Create-features" data-toc-modified-id="Create-features-41"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Create features</a></div><div class="lev3 toc-item"><a href="#Split-into-train-and-test-set" data-toc-modified-id="Split-into-train-and-test-set-411"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>Split into train and test set</a></div><div class="lev3 toc-item"><a href="#Classify" data-toc-modified-id="Classify-412"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>Classify</a></div><div class="lev3 toc-item"><a href="#Assess-our-results" data-toc-modified-id="Assess-our-results-413"><span class="toc-item-num">4.1.3&nbsp;&nbsp;</span>Assess our results</a></div><div class="lev2 toc-item"><a href="#As-a-comparison-with-a-meaningless-dataset" data-toc-modified-id="As-a-comparison-with-a-meaningless-dataset-42"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>As a comparison with a meaningless dataset</a></div><div class="lev1 toc-item"><a href="#Neural-network-solution" data-toc-modified-id="Neural-network-solution-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Neural network solution</a></div>

# <a id='top'></a>

# Source: 
# https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
# 
# **Abstract:** This data set contains electricity consumption of 370 points/clients.
# 
# **Data Set Information:**
# 
# Data set has no missing values.    
# Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.     
# Each column represent one client. Some clients were created after 2011. In these cases consumption were considered zero.     
# All time labels report to Portuguese hour. However all days present 96 measures (24*4). Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. Every year in October time change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours. 
# 
# 
# **Attribute Information:**
# * Data set were saved as txt using csv format, using semi colon (;). 
# * First column present date and time as a string with the following format 'yyyy-mm-dd hh:mm:ss' 
# * Other columns present float values with consumption in kW 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import data

# ## If you have got the data stored locally on your computer
# Otherwise see further below

# In[ ]:


path = '' # fill in correct data path here
df = pd.read_csv(path + '/ElectricityLoad_Workshop.csv',index_col=0,parse_dates=[0])
df.head(3)


# ## If working with colab
# You will need to download the data from the github repository, and upzip it.   
# Colab can mount your Google Drive, so you will need to copy the data file you want to work with onto your Google Drive.

# In[ ]:


# mount you Google Drive --> you will need to click on the link that will come up 
# and type in the authentication code that will be generated
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# now read in data
path = '/content/drive/My Drive/your_directory_for_this_workshop' # adjust the path as needed
df = pd.read_csv(path + '/ElectricityLoad_Workshop.csv',index_col=0,parse_dates=[0])
df.head(3)


# # Exploratory Data Analysis

# In[ ]:


df.shape


# In[ ]:


df.sample(3)


# In[ ]:


df.describe().round(decimals=2)


# In[ ]:


df.info()


# In[ ]:


print('The data runs from', min(df.index), 'to', max(df.index))


# In[ ]:


# how many data measurements do we have?
df.shape[0]


# In[ ]:


# how many columns = user entries?
df.shape[1]


# ## Check data index
# Do we have consecutive dates, or are some dates missing?

# In[ ]:


# step 1: look at datetimes
df.index


# In[ ]:


# step 2: compute difference between each item and the one after --> creates an array of nanosecond differences
np.diff(df.index)


# In[ ]:


# step 3: how many unique differences do we have in that array?
# ideally one one, i.e. all rows are 'equidistant'
np.unique(np.diff(df.index))


# ## Missing/null values

# In[ ]:


df.isna().sum()


# In[ ]:


# see all values
list(df.isna().sum())


# In[ ]:


# too long to read
np.unique(df.isna().sum())


# ## Visualise some sequences

# In[ ]:


# first item
df.iloc[:,0].plot(figsize=(15,10),marker='*')


# In[ ]:


# last item
df.iloc[:,-1].plot(figsize=(15,10),marker='*')


# In[ ]:


# zoom in on a subset of time - one year
start_date = pd.Timestamp('2012-01-01 00:00:00')
end_date = pd.Timestamp('2012-12-31 00:00:00')

df_subset = df[start_date:end_date]
df_subset.iloc[:,100].plot(figsize=(15,10),marker='*')


# In[ ]:


# zoom in on a subset of time - one day
start_date = pd.Timestamp('2012-01-01 00:00:00')
end_date = pd.Timestamp('2012-01-02 00:00:00')

df_subset = df[start_date:end_date]
df_subset.iloc[:,100].plot(figsize=(15,10),marker='*')


# In[ ]:


# zoom in on a subset of time : one day
start_date = pd.Timestamp('2012-01-02 00:00:00')
end_date = pd.Timestamp('2012-01-03 00:00:00')

df_subset = df[start_date:end_date]
df_subset.iloc[:,100].plot(figsize=(15,10),marker='*')


# ## Seasonality - do we have any?

# Source: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
# 
# A given time series is thought to consist of three systematic components including **level, trend, seasonality**, and one non-systematic component called **noise**.
# 
# These components are defined as follows:
# * Level: The average value in the series.
# * Trend: The increasing or decreasing value in the series.
# * Seasonality: The repeating short-term cycle in the series.
# * Noise: The random variation in the series.

# In[ ]:


# Facebook's prophet?


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot


# In[ ]:


series = df.iloc[:,100]


# In[ ]:


series.plot(figsize=(10,5))


# In[ ]:


# let's read the documentation for this command
get_ipython().run_line_magic('pinfo', 'seasonal_decompose')


# In[ ]:


# additive seasonality?
result = seasonal_decompose(series, model='additive',freq=1)
#print(result.trend)
#print(result.seasonal)
#print(result.resid)
#print(result.observed)
result.plot()
pyplot.show()


# There seems to be no additive seasonality in this time series example, and nothing attributed to random noise.

# In[ ]:


# multiplicative seasonality?

# multiplicative seasonality canot handle values of 0
result = seasonal_decompose(series+0.001, model='multiplicative',freq=1)
result.plot()
pyplot.show()


# There seems to be no multiplicative seasonality in this time series example, and nothing attributed to random noise.

# ---
# NOTE: There are other, (perhaps more sophisticated ways) to detect and deal with seasonality --> search online

# ## Group users by usage?

# In[ ]:


df.sum(axis=0).hist(bins=100)


# We can see several groups of usage.   
# There moght be a group with overall usage <= 100, and one above, but not clear.

# ## Downsample for better visualisation
# 
# Some info on date offset: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

# In[ ]:


get_ipython().run_line_magic('pinfo', 'df.resample')


# In[ ]:


# df_hourly = df.resample(axis=0,rule='H').mean()
# df_hourly.shape


# In[ ]:


# df_daily = df.resample(axis=0,rule='1d').mean()
# df_daily.shape


# In[ ]:


df_weekly = df.resample(axis=0,rule='W').mean()
df_weekly.shape


# In[ ]:


df_weekly.plot(figsize=(15,10))


# ## Other things that can be done during preprocessing
# 
# * standardise data (make mean == 0 and standard deviation ==1)
# * re-scale data (change values to be between a given min and max value)
# * check on null values and forward fill (ffill) or backward fill, depending on context
# * if you have any missing rows, i.e. time stamps, then create to make equidistant time steps
# * apply some signal processing, if that might be applicable, for instance, low-band filter
# * detect and remove data items in the signal that are not relevant, for insance, when a machine was out of order
# * detect outliers and investigate / correct / drop

# There are implementations for all kinds of pre-processing tasks within the pandas and scikit-learn packages. See documentation and tutorials on   
# https://scikit-learn.org/stable/documentation.html   
# and   
# https://pandas.pydata.org/pandas-docs/stable/#

# # Preparation for Classification

# ## Load labels

# In[ ]:


# if loading from local directory
path = '' # fill in correct data path here
labels_df = pd.read_csv(path + '/ElectricityLoad_Workshop_Labels.csv',index_col=[0])

labels_df.shape


# In[ ]:


# if loading from google drive
path = '/content/drive/My Drive/your_directory_for_this_workshop' # adjust the path as needed
df = pd.read_csv(path + '/ElectricityLoad_Workshop.csv',index_col=0,parse_dates=[0])

labels_df.shape


# In[ ]:


labels_df.head(2)


# In[ ]:


# how many different labels do we have?
labels_df.labels.unique()


# # Random forest
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html   
# 
# 'A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.'
# 
# Input has got to be one data entry == one array of feature values

# ## Create features
# 
# We will use each column of the data, i.e. each individual user, as an input data entry

# In[ ]:


# we need column-wise values as arrays
X = []

for one_col in df.columns:
    X.append(df.iloc[:][one_col].values)


# In[ ]:


len(X)


# In[ ]:


X[0]


# In[ ]:


len(X[0])


# ### Split into train and test set

# In[ ]:


y = labels_df.labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


len(X_train), len(X_test), len(y_train), len(y_test)


# In[ ]:


y_train


# In[ ]:


unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[ ]:


unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# ### Classify

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# create a classifier instance
clf_full_timeseries = RandomForestClassifier(n_estimators=100, random_state=0)


# In[ ]:


# fit our training data
clf_full_timeseries.fit(X_train, y_train)  


# In[ ]:


y_predictions = clf_full_timeseries.predict(X_test)
print(y_predictions)


# ### Assess our results

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, y_predictions, target_names=['class 0', 'class 1']))


# **Accuracy** is the answer to 'What proportion of items do we assign to the correct class?'  
# The next three measures help to interpret the results with focus on one class ('Positive class') vs. the other class or classes ('Negative class').   
# **Precision** answers the question 'What proportion of items predicted to be of Positive class are actually Positives?'   
# **Recall** is the answer to 'What proportion of Postive items has been correctly assigned to the Positive class?'   
# **F1-score** is the harmonic mean of Precision and Recall, calculated by 2* ((precision * recall)/(precision + recall))   
# Read more on https://en.wikipedia.org/wiki/F1_score

# In[ ]:


y_predictions


# In[ ]:


y_test


# ---
# Discussion: this went surprisingly well! (We don't expect 100% performance; that is actually something to be sceptical about)   
# Can we investigate what the features were that the classifier used?

# In[ ]:


len(clf_full_timeseries.feature_importances_)


# In[ ]:


print(clf_full_timeseries.feature_importances_)


# In[ ]:


np.where(clf_full_timeseries.feature_importances_ != 0)


# In[ ]:


len(np.where(clf_full_timeseries.feature_importances_ != 0)[0])


# In[ ]:


clf_full_timeseries.feature_importances_[np.where(clf_full_timeseries.feature_importances_ != 0)[0]]


# ## As a comparison with a meaningless dataset

# In[ ]:


df = pd.read_csv(path + '/ElectricityLoad_Workshop_NonMeaningfulDataset.csv',index_col=0,parse_dates=[0])

# we need column-wise values as arrays
X = []

for one_col in df.columns:
    X.append(df.iloc[:][one_col].values)
    
y = labels_df.labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# create a classifier instance
clf_full_timeseries = RandomForestClassifier(n_estimators=100, random_state=0)

# fit our training data
clf_full_timeseries.fit(X_train, y_train)  

y_predictions = clf_full_timeseries.predict(X_test)

print(classification_report(y_test, y_predictions, target_names=['class 0', 'class 1']))


# --- 
# The modelling and classification approach with a dataset that does not have any inherent patterns shows results that are around 50% accuracy. Which means in 50% of cases we assign the correct class, which is equivalent to 'by chance'.   
# This is a result to be expected for a dataset without actual patterns in data. (Unless one chose an unsuitable modelling approach or feature selection...)      
# 
# Compared to the results with our previous dataset, we can conclude that the same approach as in the last setup, but with different data, did find patterns in the data. The excellent performance is likely due to the fact that the dataset was 'crafted' for the tutorial in a way that had stark differences within the two classes' data ranges, which was picked up very well by the random forest classifier.

# # Neural network solution
# 
# We will not have time to cover solution approaches for time series data with neaural network architectures.
# 
# There are some very good step-by-step tutorials out there for this, take a look at:
# * 'How to Use the TimeseriesGenerator for Time Series Forecasting in Keras' https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
# * Deep Learning for time series tutorials https://machinelearningmastery.com/category/deep-learning-time-series/
