import numpy as np
import pandas as pd
import os

#import and shape data
cwd = os.getcwd()
datadir = cwd + os.sep + 'data' + os.sep
tweet_times = pd.read_csv(datadir + 'varol-2017-tweets.csv', encoding = 'cp1252', header = 0)
tweet_times = tweet_times[['user_id','created_at']].dropna()
tweet_times['datetime'] = pd.to_datetime(tweet_times['created_at'])

#initialize dicts for data
interarrival_times = {}
means = {}
std_devs = {}

#Find unique IDs, compute interarrival times for each, store in dicts
ids = tweet_times['user_id'].unique()
g = tweet_times.groupby(['user_id'])

for uid in ids:
    user_tweets = g.get_group(uid)
    user_interarrival = (user_tweets['datetime'].shift() - user_tweets['datetime']).astype('timedelta64[s]')
    interarrival_times[uid] = user_interarrival
    means[uid] = np.nanmean(user_interarrival.as_matrix())
    std_devs[uid] = np.nanstd(user_interarrival.as_matrix())


#Export to timing.csv    
df1 = pd.DataFrame.from_dict(means, orient = 'index')
df2 = pd.DataFrame.from_dict(std_devs, orient = 'index')

export = pd.DataFrame.merge(df1, df2, left_index = True, right_index = True )
export.columns = ['interarrival_mean', 'interarrival_std_dev']
export.index.name = 'user_id'
export.to_csv(datadir + 'timing.csv')
