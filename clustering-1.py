# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 23:36:39 2023

@author: Brent
"""

########################################################################################################
####                                                                                               #####
####                                        Clustering                                             #####
####                                                                                               #####
########################################################################################################
import pandas as pd
import os

#os.chdir("C:\\Users\\Erik\\Documents\\Brent\\USB-station\\statsthesis")
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata")


#◘df = pd.read_csv(r'dataNLPTopic.csv')
dfall = pd.read_csv(r'alldataz_sum2.csv')
aa = df.head(1000)





###########################################
##      standerdize
###########################################

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Specify the columns to standardize
cols_to_standardize = list(aa.columns[19:30]) + list(aa.columns[-57:])
cols_to_standardize = list(df.columns[17:28]) + list(df.columns[-57:])

# Standardize the columns
aa[cols_to_standardize] = scaler.fit_transform(aa[cols_to_standardize])

df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

topicsd = df['topic_1'].std
topicsd = df['topic_1'].std()

###########################################
##      subset for clustering
###########################################

## topics top 18:
# 1,12,36,41,
# 31,16,39,10,51
# 40,27,3,21,26
# 56,20,14,55


threshold = 0

subsettopic1 = df[df['topic_1'] > threshold]


subsettopic12 = df[df['topic_12'] > threshold]
subsettopic36 = df[df['topic_36'] > threshold]
subsettopic41 = df[df['topic_41'] > threshold]
subsettopic31 = df[df['topic_31'] > threshold]
subsettopic16 = df[df['topic_16'] > threshold]

subsettopic39 = df[df['topic_39'] > threshold]
subsettopic10 = df[df['topic_10'] > threshold]
subsettopic51 = df[df['topic_51'] > threshold]
subsettopic40 = df[df['topic_40'] > threshold]
subsettopic27 = df[df['topic_27'] > threshold]
subsettopic3 = df[df['topic_3'] > threshold]
subsettopic21 = df[df['topic_21'] > threshold]
subsettopic26 = df[df['topic_26'] > threshold]
subsettopic56 = df[df['topic_56'] > threshold]
subsettopic20 = df[df['topic_20'] > threshold]
subsettopic14 = df[df['topic_14'] > threshold]
subsettopic55 = df[df['topic_55'] > threshold]




subsettopic1.to_csv('subsettopic1.csv')
subsettopic12.to_csv('subsettopic12.csv')
subsettopic36.to_csv('subsettopic36.csv')
subsettopic41.to_csv('subsettopic41.csv')
subsettopic31.to_csv('subsettopic31.csv')
subsettopic16.to_csv('subsettopic16.csv')


subsettopic.to_csv('subsettopic.csv')
subsettopic.to_csv('subsettopic.csv')
subsettopic.to_csv('subsettopic.csv')
subsettopic.to_csv('subsettopic.csv')
subsettopic.to_csv('subsettopic.csv')
subsettopic.to_csv('subsettopic.csv')



a2a = subsettopic1.head(30000)
print(a2a)

df2 = subsettopic1
df2 = a2a

a2a.to_csv('df2.csv')

###########################################
##      clustering
###########################################
import numpy as np
import hdbscan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



################ old
dat = subsettopic1.iloc[:, 21:32]
aab = dat.head(10)
dat = dat.head(30000)
clust = hdbscan.HDBSCAN(min_cluster_size = 5, gen_min_span_tree=True, )
clust_lab = clust.fit_predict(dat)



################ old






average_d = []
prev_steps = 1 # 24hours * 20 minutes -> 72 timesteps per hour


for current_step in sorted(df2['Rank'].unique()):

    # Select tweets in this step and the previous 71 steps about the given topic
    mask = ((df2['Rank'] >= current_step - prev_steps) & (df2['Rank'] <= current_step))
    relevant_tweets = df2.loc[mask].copy()
    
    # Select tweets in this step about the given topic
    mask_current_step = (df2['Rank'] == current_step)
    current_step_tweets = df2.loc[mask_current_step].copy()
    
    # Perform HDBSCAN clustering on the emotion variables of relevant_tweets
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(5))
    
    cluster_data = relevant_tweets.iloc[:, 21:32]
    
    # Drop rows with missing data
    cluster_data.dropna(inplace=True)
    cluster_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(cluster_data.isnull().sum())

    # Print shape and type of the data
    print(f"Shape of the data: {cluster_data.shape}, Type of the data: {type(cluster_data)}")
    print(cluster_data.describe())
    print(cluster_data.info())
    cluster_data = cluster_data.astype(np.float64)

    cluster_labels = clusterer.fit_predict(cluster_data)

    
    # Get the probabilities of belonging to a cluster.
    probabilities = clusterer.probabilities_
    
    # Add the cluster labels and probabilities to the relevant_tweets DataFrame
    relevant_tweets['Cluster'] = cluster_labels
    relevant_tweets['Score'] = probabilities
    
    # Find the largest cluster
    largest_cluster = relevant_tweets['Cluster'].value_counts().idxmax()
    
    # Get the typical emotional pattern (average emotion scores) of this cluster
    typical_pattern = relevant_tweets[relevant_tweets['Cluster'] == largest_cluster].iloc[:,21:32].mean()
    
    # Select the data of the current step from the largest cluster
    current_step_cluster_data = current_step_tweets[current_step_tweets['Cluster'] == largest_cluster].iloc[:,21:32]
    
    # Compute the Euclidean distances of each point in the current_step_cluster_data to the typical_pattern
    distances = np.sqrt(((current_step_cluster_data - typical_pattern)**2).sum(axis=1))
    
    # Compute the average distance
    average_distance = distances.mean()
    
    # Store the average distance
    average_d.append(average_distance)
    
    # store average distances in df
    df_avg_d = pd.DataFrame({'Rank': sorted(df2['Rank'].unique())[prev_steps:],
                            'avg_d': average_d
})





#### DBSCAN

import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import numpy as np
import time


# Load the data
df2 = pd.read_csv('df2.csv')



subsettopic1 = subsettopic1.sort_values('created_at')
a2a = subsettopic1.head(300000)

print(a2a)
df2 = subsettopic1
df2 = a2a


df2 = subsettopic31


average_d = []
num_clusters = []

prev_steps = 71 # 24hours * 20 minutes -> 72 timesteps per hour   or past 6 hours = 17 (3*6 -1) 
start_time = time.time()
for current_step in sorted(df2['Rank'].unique()):
    
    # Select tweets in this step and the previous 71 steps about the given topic
    mask = ((df2['Rank'] >= current_step - prev_steps) & (df2['Rank'] <= current_step))
    relevant_tweets = df2.loc[mask].copy()
    
    # Perform DBSCAN clustering on the emotion variables of relevant_tweets
    clusterer = DBSCAN(eps=0.5, min_samples=5) # You may need to adjust these parameters
    
    cluster_data = relevant_tweets.iloc[:, 21:32]
    
    # Drop rows with missing data
    cluster_data.dropna(inplace=True)
    cluster_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    cluster_data = cluster_data.astype(np.float64)
    
    cluster_labels = clusterer.fit_predict(cluster_data)
    
    # Add the cluster labels to the relevant_tweets DataFrame
    relevant_tweets['Cluster'] = cluster_labels
    
    # Select tweets in this step about the given topic
    mask_current_step = (df2['Rank'] == current_step)
    current_step_tweets = df2.loc[mask_current_step].copy()
    
    # Add the 'Cluster' column to the current_step_tweets DataFrame
    current_step_tweets['Cluster'] = relevant_tweets.loc[mask_current_step, 'Cluster']
    
    # Find the largest cluster
    largest_cluster = relevant_tweets['Cluster'].value_counts().idxmax()
    
    # Get the typical emotional pattern (average emotion scores) of this cluster
    typical_pattern = relevant_tweets[relevant_tweets['Cluster'] == largest_cluster].iloc[:,21:32].mean()
    
    # Select the data of the current step from the largest cluster
    current_step_cluster_data = current_step_tweets[current_step_tweets['Cluster'] == largest_cluster].iloc[:,21:32]
    
    # Compute the Euclidean distances of each point in the current_step_cluster_data to the typical_pattern
    distances = distance.cdist(current_step_cluster_data, typical_pattern.values.reshape(1,-1), 'euclidean')
    
    # Compute the number of unique clusters (excluding noise, which is labeled as -1 by DBSCAN)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    # Store the number of clusters
    num_clusters.append(n_clusters)
    
    # Compute the average distance
    if len(distances) > 0:
        average_distance = distances.mean()
    else:
        average_distance = np.nan
    
    # Store the average distance
    average_d.append(average_distance)
    
# Store average distances in df
df_avg_d = pd.DataFrame({'Rank': sorted(df2['Rank'].unique()),
                        'avg_d': average_d})
df_avg_d['num_clusters'] = num_clusters



end_time = time.time()
execution_time = end_time - start_time

execution_time






df_avg_d_topic31 = df_avg_d
df_avg_d.to_csv('df_avg_d_topic31.csv')
cluster_data.to_csv('cluster_data_topic31.csv')
cluster_labels_topic31 = cluster_labels
average_d_topic31 = average_d
distances_topic31 = distances

cluster_labelsDF = cluster_labels.df 




############################################################
## reset varibales
# List all variables you want to keep
variables_to_keep = ['a2a', 'df','df2','subsettopic1']

# Get a dictionary of your current global variables
globals_ = globals().copy()

# Delete all variables except the ones you want to keep
for var in globals_:
    if var not in variables_to_keep and not var.startswith('_'):
        del globals()[var]
        
##########################################################     
        
        
        


###############################################################################•
###                                 analysis
###############################################################################•

cluster_data_topic1 = pd.read_csv(r'cluster_data_topic1.csv')
cluster_data_topic12 = pd.read_csv(r'cluster_data_topic12.csv')
cluster_data_topic31 = pd.read_csv(r'cluster_data_topic31.csv')
cluster_data_topic36 = pd.read_csv(r'cluster_data_topic36.csv')
cluster_data_topic41 = pd.read_csv(r'cluster_data_topic41.csv')

df_avg_d_topic1 = pd.read_csv(r'df_avg_d_topic1.csv')
df_avg_d_topic12 = pd.read_csv(r'df_avg_d_topic12.csv')
df_avg_d_topic31 = pd.read_csv(r'df_avg_d_topic31.csv')
df_avg_d_topic36 = pd.read_csv(r'df_avg_d_topic36.csv')
df_avg_d_topic41 = pd.read_csv(r'df_avg_d_topic41.csv')


## plot

import matplotlib.pyplot as plt


plt.plot(df_avg_d['Rank'], df_avg_d['avg_d'], marker = 'o', linestyle= '-' )


###########################
#### this is full data
plt.figure(figsize=(10,6))
plt.plot(df_avg_d_topic31['Rank'], df_avg_d_topic31['avg_d'], marker = 'o', linestyle= '-' )
plt.ylabel("Distance to dominant emotional pattern")
plt.xlabel("Time steps")
plt.title('Topic 31', fontdict=None, loc='center', pad=None)
plt.show()



###########################
#### this is smooth data

# Calculate the moving average
window_size = 12
df_avg_d_topic31['avg_d_smooth'] = df_avg_d_topic31['avg_d'].rolling(window=window_size).mean()

# Plot the original data and the smoothed data
plt.figure(figsize=(10,6))
#plt.plot(df_avg_d_topic31['Rank'], df_avg_d['avg_d'], marker='o', linestyle='-', alpha=0.5, label='Original')
plt.plot(df_avg_d_topic31['Rank'], df_avg_d_topic31['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
plt.ylabel("Distance to dominant emotional pattern")
plt.xlabel("Time steps")
plt.title('Topic 31', fontdict=None, loc='center', pad=None)
plt.show()


# Plot the original data and the smoothed data
plt.figure(figsize=(10,6))
plt.plot(df_avg_d_topic31['Rank'], df_avg_d_topic31['avg_d'], marker='o', linestyle='-', alpha=0.5, label='Original')
plt.plot(df_avg_d_topic31['Rank'], df_avg_d_topic31['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
plt.ylabel("Distance to dominant emotional pattern")
plt.xlabel("Time steps")
plt.title('Topic 31', fontdict=None, loc='center', pad=None)
plt.show()




##### function to compute the data above for all datasets


def process_and_plot(df, topic_name, window_size=12):
    # plot full data
    plt.figure(figsize=(10,6))
    plt.plot(df['Rank'], df['avg_d'], marker = 'o', linestyle= '-' )
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.show()

    # smooth data
    df['avg_d_smooth'] = df['avg_d'].rolling(window=window_size).mean()

    # plot smooth data
    plt.figure(figsize=(8,6))
    plt.plot(df['Rank'], df['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()

    # plot both original and smooth data
    plt.figure(figsize=(10,6))
    plt.plot(df['Rank'], df['avg_d'], marker='o', linestyle='-', alpha=0.5, label='Original')
    plt.plot(df['Rank'], df['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()

# you can call this function on any dataframe like this:

#process_and_plot(df_avg_d, 'Topic 0')
process_and_plot(df_avg_d_topic1, 'Topic 1')
process_and_plot(df_avg_d_topic12, 'Topic 12')
process_and_plot(df_avg_d_topic31, 'Topic 31')
process_and_plot(df_avg_d_topic41, 'Topic 41')


## new
def process_and_plot(df, topic_name, window_size=6):
    # Function to add vertical lines and annotations
    def add_vertical_lines_and_annotations():
        plt.axvline(x=1213, color='k', linestyle='--')
        plt.text(1213, df['avg_d'].min(), 'George Floyd', rotation=90, verticalalignment='bottom')
        
        plt.axvline(x=1657, color='k', linestyle='-')
        plt.text(1657, df['avg_d'].min(), 'June', rotation=90, verticalalignment='bottom', horizontalalignment='right')
        
        plt.axvline(x=3817, color='k', linestyle='-')
        plt.text(3817, df['avg_d'].min(), 'July', rotation=90, verticalalignment='bottom', horizontalalignment='right')

    # plot full data
    plt.figure(figsize=(10,6))
    plt.plot(df['Rank'], df['avg_d'], marker = 'o', linestyle= '-' )
    add_vertical_lines_and_annotations()
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.show()

    # smooth data
    df['avg_d_smooth'] = df['avg_d'].rolling(window=window_size).mean()

    # plot smooth data
    plt.figure(figsize=(8,6))
    plt.plot(df['Rank'], df['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
    add_vertical_lines_and_annotations()
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()

    # plot both original and smooth data
    plt.figure(figsize=(10,6))
    plt.plot(df['Rank'], df['avg_d'], marker='o', linestyle='-', alpha=0.5, label='Original')
    plt.plot(df['Rank'], df['avg_d_smooth'], marker='', linestyle='-', color='r', label='Smoothed')
    add_vertical_lines_and_annotations()
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()

# you can call this function on any dataframe like this:

#process_and_plot(df_avg_d, 'Topic 0')
process_and_plot(df_avg_d_topic1, 'Topic 1')
process_and_plot(df_avg_d_topic12, 'Topic 12')
process_and_plot(df_avg_d_topic31, 'Topic 31')
process_and_plot(df_avg_d_topic41, 'Topic 41')































###############################
##  
def merge_and_rename(df, topic_name):
    # Merge 'avg_d' column into 'awhole' based on the 'Rank' column
    awhole[f'avg_d_{topic_name}'] = df.set_index('Rank')['avg_d']
    # Reset the index to avoid duplicate index entries
    awhole.reset_index(inplace=True)

# Merge and rename the 'avg_d' columns for each dataset
merge_and_rename(df_avg_d, 'Topic0')
merge_and_rename(df_avg_d_topic1, 'Topic1')
merge_and_rename(df_avg_d_topic12, 'Topic12')
merge_and_rename(df_avg_d_topic31, 'Topic31')
merge_and_rename(df_avg_d_topic41, 'Topic41')



# Assuming 'awhole' is the dataset we have been working with, and it has the 'Rank' column as the index.

# Update 'awhole' with 'avg_d' columns from each dataset
awhole['avg_d_topic0'] = pd.to_numeric(df_avg_d_topic1.set_index('Rank')['avg_d'], errors='coerce')
awhole['avg_d_topic1'] = pd.to_numeric(df_avg_d_topic1.set_index('Rank')['avg_d'], errors='coerce')
awhole['avg_d_topic12'] = pd.to_numeric(df_avg_d_topic12.set_index('Rank')['avg_d'], errors='coerce')
awhole['avg_d_topic31'] = pd.to_numeric(df_avg_d_topic31.set_index('Rank')['avg_d'], errors='coerce')
awhole['avg_d_topic41'] = pd.to_numeric(df_avg_d_topic41.set_index('Rank')['avg_d'], errors='coerce')

a2awhole = awhole.head(10)



###################### now use the complete dataset:
    
import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot(awhole, topic_name, window_size=71):
    # plot full data
    plt.figure(figsize=(10, 6))
    plt.plot(awhole.index, awhole[f'avg_d_{topic_name}'], marker='o', linestyle='-')
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.show()

    # smooth data
    awhole[f'avg_d_{topic_name}_smooth'] = awhole[f'avg_d_{topic_name}'].rolling(window=window_size).mean()
    awhole['Count_smooth'] = awhole['Count'].rolling(window=window_size).mean()

    # plot smooth data
    plt.figure(figsize=(10, 6))
    plt.plot(awhole.index, awhole[f'avg_d_{topic_name}_smooth'], marker='', linestyle='-', color='r', label='Smoothed avg_d')
    plt.plot(awhole.index, awhole['Count_smooth'], marker='', linestyle='-', color='g', label='Smoothed Count')
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()

    # plot both original and smooth data
    plt.figure(figsize=(10, 6))
    plt.plot(awhole.index, awhole[f'avg_d_{topic_name}'], marker='o', linestyle='-', alpha=0.5, label='Original avg_d')
    plt.plot(awhole.index, awhole[f'avg_d_{topic_name}_smooth'], marker='', linestyle='-', color='r', label='Smoothed avg_d')
    plt.plot(awhole.index, awhole['Count_smooth'], marker='', linestyle='-', color='g', label='Smoothed Count')
    plt.ylabel("Distance to dominant emotional pattern")
    plt.xlabel("Time steps")
    plt.title(topic_name, fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.show()


process_and_plot(awhole, 'topic0')  # Replace 'Topic 1' with the desired topic name


######################### correlation

# Select the columns of interest
columns_of_interest = ['avg_d_topic0','avg_d_topic1','avg_d_topic12','avg_d_topic41', 'avg_d_topic31', 'Count', 'public_metrics.like_count', 'public_metrics.reply_count', 'public_metrics.retweet_count']

# Compute the correlation matrix
correlation_matrix = awhole[columns_of_interest].corr()
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


########################## autocorrelation

import pandas as pd


# Select the columns of interest
columns_of_interest = ['avg_d_topic1','avg_d_topic12','avg_d_topic41', 'avg_d_topic31', 'Count', 'count_topic1','public_metrics.like_count', 'public_metrics.reply_count', 'public_metrics.retweet_count']

columns_of_interest = ['avg_d_topic1','avg_d_topic12','avg_d_topic41', 'avg_d_topic31', 'Count', 'count_topic1','count_topic12','count_topic41','count_topic31']


# Calculate the autocorrelation with a lag of 1 for each column
lag_1_autocorrelation = {}
for column in columns_of_interest:
    lagged_column = awhole[column].shift(1)
    autocorr = awhole[column].corr(lagged_column)
    lag_1_autocorrelation[column] = autocorr

for column, autocorr in lag_1_autocorrelation.items():
    print(f"Lag-1 Autocorrelation for {column}: {autocorr}")


####################### cross-lagged effects


# Calculate the autocorrelation with different lags for each pair of columns
autocorrelations = pd.DataFrame()
for column1 in columns_of_interest:
    for column2 in columns_of_interest:
        for lag in range(1, 6):  # Calculate autocorrelation for lags 1 to 5 (can adjust the range as needed)
            lagged_column1 = awhole[column1].shift(lag)
            autocorr = awhole[column2].corr(lagged_column1)
            autocorrelations.loc[f"Lag {lag}", f"{column1} -> {column2}"] = autocorr

# The 'autocorrelations' DataFrame will contain the autocorrelation values for each pair of columns at different lags.
print(autocorrelations)


plt.figure(figsize=(10, 8))
sns.heatmap(autocorrelations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Autocorrelation between Variables at Different Lags")
plt.show()




############ cross lagged effects in batches:
    
columns_of_interest = ['avg_d_topic0','avg_d_topic1', 'avg_d_topic31', 'Count', 'public_metrics.like_count', 'public_metrics.reply_count', 'public_metrics.retweet_count']

# Define the 'Rank' ranges for each batch
index_ranges = [(1300, 1800), (1800, 2300), (2300,2800),(2800,3300),(3300, 3800),(3800, 4300)]

# Calculate the autocorrelation for each batch
autocorrelations_by_batch = {}
for i, (start, end) in enumerate(index_ranges, 1):
    # Select data for the current batch
    batch_data = awhole.loc[start:end]

    # Calculate the autocorrelation for each pair of columns at different lags
    autocorrelations = pd.DataFrame()
    for column1 in columns_of_interest:
        for column2 in columns_of_interest:
            for lag in range(1, 6):  # Calculate autocorrelation for lags 1 to 5 (you can adjust the range as needed)
                lagged_column1 = batch_data[column1].shift(lag)
                autocorr = batch_data[column2].corr(lagged_column1)
                autocorrelations.loc[f"Lag {lag}", f"{column1} -> {column2}"] = autocorr

    # Store the autocorrelations for the current batch
    autocorrelations_by_batch[f"Batch {i} (Index {start} - {end})"] = autocorrelations

# The 'autocorrelations_by_batch' dictionary will contain the autocorrelations for each batch.


### find the optimal negative one
autocorrelations_by_batch = {}
for i, (start, end) in enumerate(index_ranges, 1):
    # Select data for the current batch
    batch_data = awhole.loc[start:end]

    # Calculate the autocorrelation for each pair of columns at different lags
    autocorrelations = pd.DataFrame()
    for column1 in columns_of_interest:
        for column2 in columns_of_interest:
            lag_at_negative_autocorr = None
            for lag in range(1, 101):  # Calculate autocorrelation for lags 1 to 100 (you can adjust the range as needed)
                lagged_column1 = batch_data[column1].shift(lag)
                autocorr = batch_data[column2].corr(lagged_column1)
                if autocorr < 0:
                    lag_at_negative_autocorr = lag
                    break
            autocorrelations.loc[f"Lag at negative autocorrelation", f"{column1} -> {column2}"] = lag_at_negative_autocorr

    # Store the autocorrelations for the current batch
    autocorrelations_by_batch[f"Batch {i} (Index {start} - {end})"] = autocorrelations 





### from txt to jsonl

import json

# Replace 'input.txt' with the path to your TXT file containing tweets
input_file = 'input.txt'
# Replace 'output.jsonl' with the path where you want to save the JSONL file
output_file = 'output.jsonl'

# Read the contents of the TXT file
with open(input_file, 'r', encoding='utf-8') as txt_file:
    tweets_text = txt_file.readlines()

# Parse each tweet and write it as a separate line in the JSONL file
with open(output_file, 'w', encoding='utf-8') as jsonl_file:
    for tweet_text in tweets_text:
        try:
            tweet = json.loads(tweet_text)
            jsonl_file.write(json.dumps(tweet) + '\n')
        except json.JSONDecodeError:
            # Skip lines that are not valid JSON objects (e.g., headers or empty lines)
            pass

print("Conversion from TXT to JSONL completed.")




awhole.to_csv('alldataz_sum2.csv')


## visualize

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

def plot_violin(dataframes, titles):
    plt.figure(figsize=(12, 6))
    
    for idx, df in enumerate(dataframes):
        ax = plt.subplot(1, len(dataframes), idx+1)
        
        # Data for 'First 1200' and 'Last 1200' rows
        data = [df['avg_d'].iloc[:1200], df['avg_d'].iloc[-1200:]]
        
        # Plotting the violin plots
        sns.violinplot(data=data, palette=["#add8e6", "#1e90ff"], inner=None, ax=ax)
        
        # Calculating the 95% confidence interval for 'First 1200' rows and adjusting the error bar size
        ci_low_first, ci_high_first = np.percentile(data[0], [2.5, 97.5])
        error_bar_size_first = (ci_high_first - ci_low_first) * 2
        ax.errorbar(0, np.mean(data[0]), yerr=error_bar_size_first/2, fmt='o', color='red', capsize=5)
        
        # Calculating the 95% confidence interval for 'Last 1200' rows and adjusting the error bar size
        ci_low_last, ci_high_last = np.percentile(data[1], [2.5, 97.5])
        error_bar_size_last = (ci_high_last - ci_low_last) * 2
        ax.errorbar(1, np.mean(data[1]), yerr=error_bar_size_last/2, fmt='o', color='red', capsize=5)
        
        plt.title(titles[idx])
        plt.ylabel('avg_d')
        plt.xticks([0, 1], ['First 1200', 'Last 1200'])
    
    plt.tight_layout()
    plt.show()

# Assuming you have the dataframes loaded as df_avg_d_topic1, df_avg_d_topic12, df_avg_d_topic31, and df_avg_d_topic41
# You can plot them using the function as:

# plot_violin([df_avg_d_topic1, df_avg_d_topic12], ['Topic 1', 'Topic 12


# Assuming you have the dataframes loaded as df_avg_d_topic1, df_avg_d_topic12, df_avg_d_topic


# Assuming you have the dataframes loaded as df_avg_d_topic1, df_avg_d_topic12, df_avg_d_topic31, and df_avg_d_topic41
# You can plot them using the function as:

plot_violin([df_avg_d_topic1, df_avg_d_topic12], ['Topic 1', 'Topic 12'])
plot_violin([df_avg_d_topic31, df_avg_d_topic41], ['Topic 31', 'Topic 41'])






## first 1200

def plot_violin_first_1200(dataframes, titles):
    plt.figure(figsize=(12, 6))
    
    for idx, df in enumerate(dataframes):
        ax = plt.subplot(1, len(dataframes), idx+1)
        
        # Data for 'First 1200' rows
        data_first = df['avg_d'].iloc[:1200].values
        
        # Plotting the violin plot for 'First 1200' rows
        sns.violinplot(y=data_first, color="#add8e6", ax=ax)
        
        # Calculating and plotting the 95% confidence interval for 'First 1200' rows
        ci_low_first, ci_high_first = np.percentile(data_first, [2.5, 97.5])
        ax.axhline(ci_low_first, color="red", linestyle="--")
        ax.axhline(ci_high_first, color="red", linestyle="--")
        
        plt.title(titles[idx] + " - First 1200")
        plt.ylabel('avg_d')
    
    plt.tight_layout()
    plt.show()

plot_violin_first_1200([df_avg_d_topic1, df_avg_d_topic12], ['Topic 1', 'Topic 12'])











### looks great

def plot_violin(dataframes, titles):
    plt.figure(figsize=(12, 6))
    
    for idx, df in enumerate(dataframes):
        ax = plt.subplot(1, len(dataframes), idx+1)
        
        # Data for 'First 1200' and 'Last 1200' rows
        data_first = df['avg_d'].iloc[:1200].values
        data_last = df['avg_d'].iloc[-1200:].values
        
        # Plotting the violin plots
        sns.violinplot(data=[data_first, data_last], palette=["#add8e6", "#1e90ff"], ax=ax)
        
        # Calculating and plotting the 95% confidence interval for 'First 1200' rows
        ci_low_first, ci_high_first = np.percentile(data_first, [2.5, 97.5])
        ax.axhline(ci_low_first, color="red", linestyle="--", xmin=0.05, xmax=0.45)
        ax.axhline(ci_high_first, color="red", linestyle="--", xmin=0.05, xmax=0.45)
        
        # Calculating and plotting the 95% confidence interval for 'Last 1200' rows
        ci_low_last, ci_high_last = np.percentile(data_last, [2.5, 97.5])
        #ax.axhline(ci_low_last, color="pink", linestyle="--", xmin=0.55, xmax=0.95)
        #ax.axhline(ci_high_last, color="blue", linestyle="--", xmin=0.55, xmax=0.95)
        
        plt.title(titles[idx])
        plt.ylabel('average distance to Dominant Emotional Pattern')
        plt.xticks([0, 1], ['Before George Floyd', '1-2 months after George Floyd'])
    
    plt.tight_layout()
    plt.show()

# Assuming you have the dataframes loaded as df_avg_d_topic1, df_avg_d_topic12, df_avg_d_topic31, and df_avg_d_topic41
# You can plot them using the function as:

plot_violin([df_avg_d_topic1, df_avg_d_topic12], ['Topic 1: cognitive thinking', 'Topic 12: posting about BLM'])
plot_violin([df_avg_d_topic31, df_avg_d_topic41], ['Topic 31: positivity', 'Topic 41: BLM protest'])





### now significance tests

from scipy.stats import ttest_ind, levene, t
import numpy as np

def welch_dof(x,y):
    """ Calculate the effective degrees of freedom for two samples. """
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
    return dof

def cohen_d(x, y):
    """ Calculate Cohen's d for effect size. """
    pooled_var = (len(x) * np.var(x, ddof=1) + len(y) * np.var(y, ddof=1)) / (len(x) + len(y))
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)
    return d

dataframes = [df_avg_d_topic1, df_avg_d_topic12, df_avg_d_topic31, df_avg_d_topic41]
titles = ['Topic 1', 'Topic 12', 'Topic 31', 'Topic 41']

def variance_ratio(first, last):
    """Compute the F ratio (variance ratio) for comparing variances."""
    return first.var() / last.var()

from scipy.stats import kstest, ttest_ind, levene
for idx, df in enumerate(dataframes):
    first = df['avg_d'].iloc[:1200]
    last = df['avg_d'].iloc[-1200:]
    
    # Check for NaN or infinite values
    if np.any(np.isnan(first)) or np.any(np.isnan(last)):
        print(f"{titles[idx]} has NaN values.")
    if np.any(np.isinf(first)) or np.any(np.isinf(last)):
        print(f"{titles[idx]} has infinite values.")
    
    # Handle NaN values by removing them (you can also impute them if you prefer)
    first = first.dropna()
    last = last.dropna()
    
    # Handle infinite values by removing them (you can also replace them if you prefer)
    first = first[np.isfinite(first)]
    last = last[np.isfinite(last)]
    
    # Kolmogorov-Smirnov test for normality
    _, p_value_ks_first = kstest(first, 'norm')
    _, p_value_ks_last = kstest(last, 'norm')
    
    # Test for equality of variances
    _, p_value_var = levene(first, last)
    
    # Welch's t-test
    t_stat, p_value = ttest_ind(first, last, equal_var=False)
    
    # Degrees of freedom for Welch's t-test
    dof = welch_dof(first, last)
    
    # Effect size (Cohen's d)
    d = cohen_d(first, last)
    delta = variance_ratio(first, last)

    print(f"{titles[idx]} Glass's Δ: {delta:.4f}")
    print(f"K-S test p-value (First 1200): {p_value_ks_first:.4f}")
    print(f"K-S test p-value (Last 1200): {p_value_ks_last:.4f}")
    
    print(f"{titles[idx]}:")
    print(f"Levene's test p-value: {p_value_var:.4f}")
    print(f"Welch's t-test: t({dof:.2f}) = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"Cohen's d: {d:.4f}")
    print("-" * 50)


#### assumption of normality violated

from scipy.stats import mannwhitneyu, fligner

def rank_biserial(u, n1, n2):
    """Compute rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2*u) / (n1 * n2)

def variance_ratio(first, last):
    """Compute the F ratio (variance ratio) for comparing variances."""
    return first.var() / last.var()

for idx, df in enumerate(dataframes):
    first = df['avg_d'].iloc[:1200]
    last = df['avg_d'].iloc[-1200:]
    
    # Handle NaN and infinite values
    first = first.dropna()
    last = last.dropna()
    first = first[np.isfinite(first)]
    last = last[np.isfinite(last)]
    
    # Mann-Whitney U test for comparing medians
    u_stat, p_value_median = mannwhitneyu(first, last, alternative='two-sided')
    
    # Rank-biserial correlation as effect size for Mann-Whitney U
    rbc = rank_biserial(u_stat, len(first), len(last))
    
    # Fligner-Killeen test for equality of variances
    _, p_value_var = fligner(first, last)
    
    # Glass's Δ for variance ratio
    delta = variance_ratio(first, last)

    print(f"{titles[idx]}:")
    print(f"Mann-Whitney U test p-value: {p_value_median:.4f}")
    print(f"Rank-biserial correlation: {rbc:.4f}")
    print(f"Fligner-Killeen test p-value: {p_value_var:.4f}")
    print(f"Glass's Δ: {delta:.4f}")
    print("-" * 50)



##nsign tests for time series
from scipy.stats import shapiro, ttest_ind

def check_normality_and_ttest(dataframes, titles):
    for idx, df in enumerate(dataframes):
        # Extract data for 'First 1200' and 'Last 1200' rows
        data_first = df['avg_d'].iloc[:1200]
        data_last = df['avg_d'].iloc[-1200:]

        # Shapiro-Wilk Test for normality
        stat_first, p_first = shapiro(data_first)
        stat_last, p_last = shapiro(data_last)

        print(f"\nFor {titles[idx]}:")
        print(f"Shapiro-Wilk Test for 'First 1200': Statistic={stat_first}, p-value={p_first}")
        print(f"Shapiro-Wilk Test for 'Last 1200': Statistic={stat_last}, p-value={p_last}")

        # If p-value is less than 0.05, the data is not normally distributed
        if p_first < 0.05:
            print("'First 1200' data is not normally distributed.")
        else:
            print("'First 1200' data is normally distributed.")

        if p_last < 0.05:
            print("'Last 1200' data is not normally distributed.")
        else:
            print("'Last 1200' data is normally distributed.")

        # If both series are approximately normal, proceed with the T-test
        if p_first >= 0.05 and p_last >= 0.05:
            t_stat, p_val = ttest_ind(data_first, data_last)
            print(f"T-test results: Statistic={t_stat}, p-value={p_val}")
            if p_val < 0.05:
                print("There is a significant difference between the means of 'First 1200' and 'Last 1200'.")
            else:
                print("There is no significant difference between the means of 'First 1200' and 'Last 1200'.")

check_normality_and_ttest([df_avg_d_topic1, df_avg_d_topic12], ['Topic 1', 'Topic 12'])



## bootstrapping

import numpy as np
import matplotlib.pyplot as plt

def block_bootstrap(series, block_size, num_samples):
    """Generate bootstrapped samples using block bootstrapping."""
    samples = []
    n = len(series)
    for _ in range(num_samples):
        sample = []
        while len(sample) < n:
            start_idx = np.random.randint(0, n - block_size + 1)
            sample.extend(series[start_idx:start_idx+block_size])
        samples.append(sample[:n])
    return samples

# Check for NaN values in the original data
if df_avg_d_topic1['avg_d'].isna().any() or df_avg_d_topic12['avg_d'].isna().any():
    print("Warning: NaN values detected in the data. Please handle them before proceeding.")
    # You can choose to drop NaN values or fill them
    # df_avg_d_topic1 = df_avg_d_topic1.dropna()
    # df_avg_d_topic12 = df_avg_d_topic12.dropna()

# Parameters
block_size = 50  # Size of blocks to resample
num_samples = 2000  # Number of bootstrapped samples

df_avg_d_topic1 = df_avg_d_topic1.dropna(subset=['avg_d'])
df_avg_d_topic12 = df_avg_d_topic12.dropna(subset=['avg_d'])

# Generate bootstrapped samples
bootstrap_samples_topic1 = block_bootstrap(df_avg_d_topic1['avg_d'].values, block_size, num_samples)
bootstrap_samples_topic12 = block_bootstrap(df_avg_d_topic12['avg_d'].values, block_size, num_samples)

# Compute differences in means for each pair of bootstrapped samples
differences = [np.mean(sample1) - np.mean(sample2) for sample1, sample2 in zip(bootstrap_samples_topic1, bootstrap_samples_topic12)]

# Remove any NaN values from differences
differences = [diff for diff in differences if not np.isnan(diff)]

# Plot histogram of differences
plt.hist(differences, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Bootstrapped Mean Differences')
plt.xlabel('Difference in Means')
plt.ylabel('Frequency')
plt.show()

# Compute 95% confidence interval
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
print(f"95% Confidence Interval for Difference in Means: ({ci_low:.4f}, {ci_high:.4f})")


## new bootstrapping

def block_bootstrap(data, block_size, num_samples):
    num_blocks = len(data) - block_size + 1
    bootstrap_samples = []
    
    for _ in range(num_samples):
        sample = []
        while len(sample) < len(data):
            start_idx = np.random.randint(0, num_blocks)
            block = data[start_idx:start_idx+block_size]
            sample.extend(block)
        bootstrap_samples.append(sample[:len(data)])
    
    return bootstrap_samples

def bootstrap_difference(data1, data2, block_size=10, num_iterations=10000):
    differences = []
    
    for _ in range(num_iterations):
        sample1 = block_bootstrap(data1, block_size, 1)[0]
        sample2 = block_bootstrap(data2, block_size, 1)[0]
        mean_diff = np.mean(sample1) - np.mean(sample2)
        differences.append(mean_diff)
    
    return differences


def bootstrap_difference(data1, data2, block_size=10, num_iterations=10000):    #+ this is for variance
    variance_differences = []
    
    for _ in range(num_iterations):
        sample1 = block_bootstrap(data1, block_size, 1)[0]
        sample2 = block_bootstrap(data2, block_size, 1)[0]
        variance_diff = np.var(sample1) - np.var(sample2)
        variance_differences.append(variance_diff)
    
    return variance_differences



def cohens_d_for_means(differences):
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    return mean_diff / std_diff


################ Data for topic 1
first_1200_topic1 = df_avg_d_topic1['avg_d'].iloc[:1200].dropna().values
last_1200_topic1 = df_avg_d_topic1['avg_d'].iloc[-1200:].dropna().values

differences = bootstrap_difference(first_1200_topic1, last_1200_topic1)
plt.hist(differences, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Bootstrapped variance Differences for topic 1')
plt.xlabel('Difference in variances')
plt.ylabel('Frequency')
plt.show()
# Compute 95% confidence interval
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
print(f"95% Confidence Interval for Difference in Means: ({ci_low:.4f}, {ci_high:.4f})")
d_means = cohens_d_for_means(differences)
print(f"Cohen's d for Mean Differences: {d_means:.4f}")

################ Data for topic 12
first_1200_topic12 = df_avg_d_topic12['avg_d'].iloc[:1200].dropna().values
last_1200_topic12 = df_avg_d_topic12['avg_d'].iloc[-1200:].dropna().values

differences = bootstrap_difference(first_1200_topic12, last_1200_topic12)
plt.hist(differences, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Bootstrapped variances Differences  for topic 12')
plt.xlabel('Difference in variances')
plt.ylabel('Frequency')
plt.show()
# Compute 95% confidence interval
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
print(f"95% Confidence Interval for Difference in Means: ({ci_low:.4f}, {ci_high:.4f})")
d_means = cohens_d_for_means(differences)
print(f"Cohen's d for Mean Differences: {d_means:.4f}")

################ Data for topic 41
first_1200_topic41 = df_avg_d_topic41['avg_d'].iloc[:1200].dropna().values
last_1200_topic41 = df_avg_d_topic41['avg_d'].iloc[-1200:].dropna().values

differences = bootstrap_difference(first_1200_topic41, last_1200_topic41)
plt.hist(differences, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Bootstrapped variances Differences  for topic 41')
plt.xlabel('Difference in variances')
plt.ylabel('Frequency')
plt.show()
# Compute 95% confidence interval
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
print(f"95% Confidence Interval for Difference in variances: ({ci_low:.4f}, {ci_high:.4f})")
d_means = cohens_d_for_means(differences)
print(f"Cohen's d for Mean Differences: {d_means:.4f}")
d_means = cohens_d_for_means(differences)
print(f"Cohen's d for Mean Differences: {d_means:.4f}")


################ Data for topic 1
first_1200_topic31 = df_avg_d_topic31['avg_d'].iloc[:1200].dropna().values
last_1200_topic31 = df_avg_d_topic31['avg_d'].iloc[-1200:].dropna().values

differences = bootstrap_difference(first_1200_topic31, last_1200_topic31)
plt.hist(differences, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Bootstrapped variances Differences  for topic 31')
plt.xlabel('Difference in variances')
plt.ylabel('Frequency')
plt.show()
# Compute 95% confidence interval
ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
print(f"95% Confidence Interval for Difference in Means: ({ci_low:.4f}, {ci_high:.4f})")
d_means = cohens_d_for_means(differences)
print(f"Cohen's d for Mean Differences: {d_means:.4f}")






#########################

import pandas as pd
import matplotlib.pyplot as plt

# Assuming dfall is already loaded
plt.figure(figsize=(12, 6))
plt.plot(dfall['Count'])
plt.title('Time Series of total amount of tweets per time step')
plt.xlabel('Time steps')
plt.ylabel('Amount of tweets posted about BLM')
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(dfall['count_topic1'])
plt.title('Time Series of total amount of tweets per time step')
plt.xlabel('Time steps')
plt.ylabel('Amount of tweets posted about BLM')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(dfall['count_topic41'])
plt.title('Time Series of Count')
plt.xlabel('Time steps')
plt.ylabel('Count')
plt.grid(True)
plt.show()


















