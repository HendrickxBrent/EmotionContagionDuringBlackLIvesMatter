# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 19:29:25 2022
@author: Brent
"""

## make sure you are in a new environment
conda activate thesisstats

!conda info --envs
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu112
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu112/torch_stable.html
pip install transformers
pip install tweetnlp


import tweetnlp

#RoBERTa
import concurrent.futures
import time
import pandas as pd
import torch
import transformers
from transformers import pipeline
import tensorflow as tf
import timeit
import dask.dataframe as dd
import os
import numpy as np
import re
import time
import concurrent.futures
import dask.dataframe as dd
from multiprocessing import cpu_count

cores = 10  
####
classifier = tweetnlp.load_model('topic_classification', model_name='cardiffnlp/twitter-roberta-base-emotion-multilabel-latest')
classifier.predict("I bet everything will work out in the end :)", return_probability=True)


import os
import signal
# Get the current process ID
pid = os.getpid()
# Send a signal to restart the process
os.kill(pid, signal.SIGINT)



os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata")
df = pd.read_csv('day31.csv')

df=df2
df = df.reset_index()
df = df.replace(np.nan,'')

def remove(text):
    #text = re.sub('[0-9]+', '', text)           # numbers
    text = re.sub(r"http\S+", "", text)         #links
    text = re.sub(r"www.\S+", "", text)         #links
    text = re.sub("@[A-Za-z0-9_]+","", text)    #remove @mentions
    return text

df['Tweet'] = df['text'].apply(lambda x: remove(x))


#################################################################
#### after BLM

# List of words to search for
keywords = ['blm', 'blacklives', 'georgefl', 'floyd', 'racism', 'looting', 'supremacists']

# Create a regular expression pattern that matches any of the words
pattern = '|'.join(keywords)

# Use the `str.contains()` method to create a boolean mask that is True if the "Tweet" column contains any of the words
mask = df['Tweet'].str.contains(pattern, case=False, na=False)

# Convert the boolean mask to integers, so that True becomes 1 and False becomes 0
df['blm'] = mask.astype(int)

subset_df = df[(df['blm'] == 0) & (df['lang'] == 'en')]


#################################################################




## split dataset to train in chunks
one_df1 = 750000 
two_df1 = 750001

A = 0

df1=df
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel1June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel2June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel3June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel4June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')


df1 = pd.read_csv(r'June30rest.csv')
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel5June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel6June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel7June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel8June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')


df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel9June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel10June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')


df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel11June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel12June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')





df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel13June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel14June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')


df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel15June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel16June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel17June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel18June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel19June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel20June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')





df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel21June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel22June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel23June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel24June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel25June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')

df1 = pd.read_csv(r'June30rest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel26June30.csv')  ## tot 
twoApril.to_csv('June30rest.csv')






## new split approach

df = pd.read_csv('your_data.csv')  # Load the original data

batch_size = 600000
num_batches = len(df) // batch_size  # Calculate the number of batches

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_df = df.iloc[start_idx:end_idx, :]  # Select the rows for the current batch
    batch_df.to_csv(f'batch_{i+1}.csv', index=False)  # Save the batch to a separate file






####################################################################################################
###                         withhout gpu and without multiprocessing
###
### speed benchmark: dataset 2000 rows -> 73 seconds
###                          4000 rows -> 127.21
####################################################################################################
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]

emotion_df = pd.DataFrame(index=df2.index, columns=labels)

start = timeit.default_timer()
for i, row in df2.iterrows():
    # Apply the classifier function to the Tweet in the 'Tweet' column
    result = classifier(df2.at[i, 'Tweet'])
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    # Loop through each emotion in the dictionary
    for emotion in result[0]:
        # If the emotion label exists in the new dataframe,
        # set the value for that emotion in the current row
        # to the corresponding score from the dictionary
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']
result2 = pd.concat([df2, emotion_df], axis=1)
end = timeit.default_timer()
print("Time taken:", end - start)






####################################################################################################
###                                          threading
###
### speed benchmark: dataset 2000 rows -> 37.27 seconds (I7, 10th edition, 11 cores)
###                          4000 rows -> 73.08
###                          10000 rows-> 220.31
####################################################################################################



classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]

emotion_df = pd.DataFrame(index=df2.index, columns=labels)


def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]

start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df2[['Tweet']].itertuples(index=True)

    # Map the classify_emotion function to the iterator
    # using the executor's map method
    
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    

    # Initialize the emotion dataframe with the same shape as the original dataframe
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)

    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']

    # Concatenate the original dataframe with the emotion dataframe
    result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")




####################################################################################################
###                                  Parallel processing (multiple CPU) DASK
###
### speed benchmark: dataset 2000 rows -> 39.63 (I7, 10th edition, 11 cores)
###                          4000 rows -> 74.22
###                          10000
####################################################################################################
####################################################################################################
## DASK   this is 28.9 seconds (2000)   for 4000 = 57.63


start = time.time()
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]

# Convert the Pandas dataframe to a Dask dataframe
df = dd.from_pandas(df2, npartitions=11)

# Use the map method of the Dask dataframe to apply the classify_emotion function to each tweet
results = df['Tweet'].map(classify_emotion)

# Compute the result as a Pandas dataframe
results = results.compute()

# Initialize the emotion dataframe with the same shape as the original dataframe
emotion_df = pd.DataFrame(index=df2.index, columns=labels)

# Loop through the results and populate the emotion dataframe
for i, (labels, emotions) in enumerate(results):
    for emotion in emotions:
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']

# Concatenate the original dataframe with the emotion dataframe
result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")




####################################################################################################
###                                          DASK + threading
###
### speed benchmark: dataset 2000 rows ->  37.60   (I7, 10th edition, 2*11 cores) 
###                          2000 rows ->  38.78   (I7, 10th edition, 5+6 cores) 
###                          4000 rows ->  79.45   (I7, 10th edition, 11 cores)
####################################################################################################
####################################################################################################
## DASK + threathing   this is 28.9 seconds+ multiple threathing   =  28.5 (2000)    for 4000 = 55.57   (using cpu 11 cores and intel 10th gen i7)


import concurrent.futures

start = time.time()
def classify_emotion(tweet):
    # Initialize a ThreadPoolExecutor with 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Apply the classifier function to the given tweet text
        result = classifier(tweet)
        # Get the labels (emotions) from the result dictionary
        labels = [entry['label'] for entry in result[0]]
        return labels, result[0]

# Convert the Pandas dataframe to a Dask dataframe
df = dd.from_pandas(df2, npartitions=6)

# Use the map method of the Dask dataframe to apply the classify_emotion function to each tweet
results = df['Tweet'].map(classify_emotion)

# Compute the result as a Pandas dataframe
results = results.compute()

# Initialize the emotion dataframe with the same shape as the original dataframe
emotion_df = pd.DataFrame(index=df2.index, columns=labels)

# Loop through the results and populate the emotion dataframe
for i, (labels, emotions) in enumerate(results):
    for emotion in emotions:
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']

# Concatenate the original dataframe with the emotion dataframe
result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

















CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))


############################### code############################### code    GPU acceleration test   (didn't work)
############################### code############################### code


print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
# Check if the GPU is available
if tf.test.is_gpu_available():
    # Set the device to GPU
    device = '/device:GPU:0'
    print("Running on the GPU")
else:
    # Set the device to CPU
    device = '/device:CPU:0'
    print("Running on the CPU")

tf.config.list_physical_devices('GPU')

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]

emotion_df = pd.DataFrame(index=df2.index, columns=labels)

##### with gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = timeit.default_timer()
with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
    for i, row in df2.iterrows():
        # Apply the classifier function to the text in the 'Tweet' column
        result = classifier(df2.at[i, 'Tweet'])
        # Get the labels (emotions) from the result dictionary
        labels = [entry['label'] for entry in result[0]]
        # Loop through each emotion in the dictionary
        for emotion in result[0]:
            # If the emotion label exists in the new dataframe,
            # set the value for that emotion in the current row
            # to the corresponding score from the dictionary
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
result2 = pd.concat([df2, emotion_df], axis=1)
end = timeit.default_timer()
print("Time taken:", end - start)








#########################################################################################################################################
########################################                  start new here                          #######################################

import numpy as np
import pandas as pd
#option 1
import transformers
import tensorflow
from transformers import pipeline


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)  # Get the name of the current GPU
    print(f"Using GPU: {device}")
else:
    print("Using CPU")



#☺option2
#pip install tokenizers
#pip install tweetnlp

import torch
import tweetnlp

df2 = subset_df
df2 = df2.reset_index()
df2 = df2.replace(np.nan,'')


df1 = df1.reset_index()





####
classifier = tweetnlp.load_model('topic_classification', model_name='cardiffnlp/twitter-roberta-base-emotion-multilabel-latest')
classifier.predict("I bet everything will work out in the end :)", return_probability=True)


n_cores = cpu_count()
cores = n_cores - 1   # don't use all cores if you want the basic programs of your computer to function properly
cores = 10      # when i'm working during the day on my computer
# cores = 11      # during the night when im not working

def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier.predict(tweet, return_probability=True)
    # Get the labels (emotions) from the result dictionary
    labels = list(result['probability'].keys())
    result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
    return labels, result


#CUDA_LAUNCH_BLOCKING=1





#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\earlier")
df2 = pd.read_csv('tweetsmai25_.csv')

#df2 = df2.head(1000)
df2 = df
df1=df

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('resultsday31.csv')





#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_10")
df2 = pd.read_csv('deel2June10.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_10")
df2 = pd.read_csv('deel3June10.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results3.csv')




#############################
### deel4
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel4June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results4.csv')







#############################
### deel5
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel5June2.csv')



# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''


#df1 = df2.sample(n=1000)
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results5.csv')




#############################
### deel6
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel6June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results6.csv')


#############################
### deel7
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel7June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results7.csv')


#############################
### deel8
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel8June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results8.csv')




#############################
### deel9
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel9June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results9.csv')


#############################
### deel10
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel10June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results10.csv')




#############################
### deel11
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_02")
df2 = pd.read_csv('deel11June2.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results11.csv')


############################################################################################################
############################################################################################################
############################################################################################################

###                                day 11

############################################################################################################
############################################################################################################
############################################################################################################



#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_11")
df2 = pd.read_csv('deel1June11.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')





#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_11")
df2 = pd.read_csv('deel2June11.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_11")
df2 = pd.read_csv('deel3June11.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results3.csv')




#############################
### deel4
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_04")
df2 = pd.read_csv('deel4June4.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results4.csv')







#############################
### deel5
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_04")
df2 = pd.read_csv('deel5June4.csv')



# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''


#df1 = df2.sample(n=1000)
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results5.csv')




#############################
### deel6
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_04")
df2 = pd.read_csv('deel6June4.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results6.csv')


#############################
### deel7
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_04")
df2 = pd.read_csv('deel7June4.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results7.csv')





############################################################################################################
############################################################################################################
############################################################################################################

###                                day 12

############################################################################################################
############################################################################################################
############################################################################################################



#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_12")
df2 = pd.read_csv('deel1June12.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')





#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_12")
df2 = pd.read_csv('deel2June12.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_12")
df2 = pd.read_csv('deel3June12.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results3.csv')




#############################
### deel4
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_05")
df2 = pd.read_csv('deel4June5.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results4.csv')







#############################
### deel5
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_05")
df2 = pd.read_csv('deel5June5.csv')



# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''


#df1 = df2.sample(n=1000)
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results5.csv')




#############################
### deel6
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_05")
df2 = pd.read_csv('deel6June5.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results6.csv')





############################################################################################################
############################################################################################################
############################################################################################################

###                                day 15

############################################################################################################
############################################################################################################
############################################################################################################



#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_16")
df2 = pd.read_csv('deel1June16.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')





#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_16")
df2 = pd.read_csv('deel2June16.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_16")
df2 = pd.read_csv('deel3June16.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results3.csv')




#############################
### deel4
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_13")
df2 = pd.read_csv('deel4June13.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results4.csv')







#############################
### deel5
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_06")
df2 = pd.read_csv('deel5June6.csv')



# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''


#df1 = df2.sample(n=1000)
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results5.csv')




#############################
### deel6
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_06")
df2 = pd.read_csv('deel6June6.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results6.csv')



############################################################################################################
############################################################################################################
############################################################################################################

###                                day 14

############################################################################################################
############################################################################################################
############################################################################################################



#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_14")
df2 = pd.read_csv('deel1June14.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')





#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_14")
df2 = pd.read_csv('deel2June14.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_14")
df2 = pd.read_csv('deel3June14.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results3.csv')




#############################
### deel4
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_07")
df2 = pd.read_csv('deel4June7.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results4.csv')







#############################
### deel5
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_07")
df2 = pd.read_csv('deel5June7.csv')



# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''


#df1 = df2.sample(n=1000)
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results5.csv')





























############################################################################################################
############################################################################################################
############################################################################################################

###                                day 18

############################################################################################################
############################################################################################################
############################################################################################################



#############################
### deel16
###############6##############
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel16June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results16.csv')



#############################
### deel17
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel17June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results17.csv')


#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel18June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results18.csv')

#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel20June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results20.csv')

#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel21June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results21.csv')


#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_03")
df2 = pd.read_csv('deel5June3.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')




#############################
### deel1
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_30x07_20")
df2 = pd.read_csv('deel15June30.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results15.csv')



#############################
### deel2
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_03")
df2 = pd.read_csv('deel2June3.csv')

#df2 = df2.head(1000)
df1 = df2

# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results2.csv')





#############################
### deel3
#############################
###                          
os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\thesisstatisticsdata\\tweetsMA2_06_06")
df2 = pd.read_csv('June6rest.csv')

df1 = df2

df1 = df1.dropna(subset=['Tweet'])

#df2 = df2.head(1000)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
extract = [{'label': label, 'score': score} for label, score in result['probability'].items()]
result = classifier.predict(df2.at[0, 'Tweet'], return_probability=True)
labels = list(result['probability'].keys())
#labels = [entry['label'] for entry in result[0]]
emotion_words = list(result['probability'].keys())
for word in emotion_words:
    df2[word] = ''
    
def classify_emotion(tweet):
        # Apply the classifier function to the given tweet text
        result = classifier.predict(tweet, return_probability=True)
        # Get the labels (emotions) from the result dictionary
        labels = list(result['probability'].keys())
        result = [{'label': label, 'score': score} for label, score in result['probability'].items()]
        return labels, result



start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
torch.cuda.empty_cache()
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results6.csv')

































##### with gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = timeit.default_timer()
with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
    for i, row in df2.iterrows():
        # Apply the classifier function to the text in the 'Tweet' column
        result = classifier(df2.at[i, 'Tweet'])
        # Get the labels (emotions) from the result dictionary
        labels = [entry['label'] for entry in result[0]]
        # Loop through each emotion in the dictionary
        for emotion in result[0]:
            # If the emotion label exists in the new dataframe,
            # set the value for that emotion in the current row
            # to the corresponding score from the dictionary
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
result2 = pd.concat([df2, emotion_df], axis=1)
end = timeit.default_timer()
print("Time taken:", end - start)





