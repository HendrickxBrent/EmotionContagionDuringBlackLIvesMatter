# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 15:49:58 2022

@author: Brent
"""

import pandas as pd
import glob
import os



# setting the path for joining multiple files
files = os.path.join("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata", "all*.csv")


# list of merged files returned
files = glob.glob(files)
print("Resultant CSV after joining all CSV files at a particular location...");
# joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
print(df)


os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata")

aaa = df.head(1000)
## clean data
#df2  = pd.DataFrame(df[['author.id', 'text','created_at']])


df.to_csv('data.csv')



####
import os
import glob
import pandas as pd



# Setting the path for joining multiple files
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata")
files = os.path.join("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata", "all*.csv")

# List of merged files returned
files = glob.glob(files)
print("Resultant CSV after joining all CSV files at a particular location...")

# Joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
print(df)

## Clean data
df2 = pd.DataFrame(df[['id','conversation_id','author.id', 'text','created_at','lang','public_metrics.like_count', 'public_metrics.reply_count'
                        ,'public_metrics.retweet_count','author.public_metrics.tweet_count','author.public_metrics.followers_count'
                        ,'author.public_metrics.following_count','author.verified','Tweet']])

# Create new columns for emotions
emotion_mapping = {
    'anger': 'anger',
    'anticipation': 'anticipation',
    'disgust': 'disgust',
    'fear': 'fear',
    'joy': 'joy',
    'love': 'love',
    'optimism': 'optimism',
    'pessimism': 'pessimism',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'trust': 'trust'
}

for emotion in emotion_mapping.keys():
    columns = [col for col in df.columns if emotion in col]
    df2[emotion_mapping[emotion]] = df[columns].sum(axis=1)

df2.to_csv('data.csv')




