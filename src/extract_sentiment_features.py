#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:49:48 2017

@author: stephencarrow
"""

import pandas as pd
import os
import glob
from feature_extractors.sentiment import SentimentExtractor
import sys

def main():
    
    # Load honeypot data and label
    cwd = os.getcwd()
    main_directory = '/'.join(cwd.split('/')[:-1])
    
    file_list = glob.glob(main_directory + '/data/social_honeypot_icwsm_2011/' + '*.txt')
    
    cols_dict = {"content_polluters.txt": ["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", 
                                           "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName",
                                           "LengthOfDescriptionInUserProfile"],
                 "content_polluters_followings.txt": ["UserID", "SeriesOfNumberOfFollowings"],
                 "content_polluters_tweets.txt": ["UserID", "TweetID", "Tweet", "CreatedAt"],
                 "legitimate_users.txt": ["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", 
                                           "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName",
                                           "LengthOfDescriptionInUserProfile"],
                 "legitimate_users_followings.txt": ["UserID", "SeriesOfNumberOfFollowings"],
                 "legitimate_users_tweets.txt": ["UserID", "TweetID", "Tweet", "CreatedAt"]}
    
    glb = globals()
    for each_location in file_list:
        each_file = each_location.split('/')[-1].split('.')[0]
        glb[each_file+"_df"] = pd.read_table(each_location,
           names=cols_dict[each_file+".txt"])
        
    content_polluters_tweets_df['label'] = 1
    legitimate_users_tweets_df['label'] = 0
    
    # Load ANEW dictionary
    anew_df = pd.read_csv(main_directory + '/data/Ratings_Warriner_et_al.csv',
                         header=0, index_col=0)
    anew_df.set_index('Word', inplace=True)
    
    
    #Apply sentiment feature extractors
    tweet_df = pd.concat([legitimate_users_tweets_df,
                          content_polluters_tweets_df], ignore_index=True)
        
    extractor = SentimentExtractor()
    tweets_vader = extractor.TweetsVADER(tweet_df.Tweet, tweet_df.TweetID.values)
    tweets_anew = extractor.TweetsANEW(tweet_df.Tweet, tweet_df.TweetID.values, anew_df)
    
    tweet_df = tweet_df.merge(tweets_vader, on="TweetID")
    tweet_df = tweet_df.merge(tweets_anew, on="TweetID")
    
    # Summarise scores using distributions at user level
    sentiment_dist_df = tweet_df[['UserID','compound','neg','neu','pos',
                                  'valence','arousal','dominance']].groupby(
                                  'UserID').describe()
        
    sentiment_dist_df = sentiment_dist_df.drop([('compound','count'),('neg','count'),
                                                ('neu','count'),('pos','count'),
                                                ('valence','count'),('arousal','count'),
                                                ('dominance','count')], 1)
        
    colnames = [' '.join(col).strip() for col in sentiment_dist_df.columns.values]
    sentiment_dist_df.columns = sentiment_dist_df.columns.droplevel(0)
    sentiment_dist_df.columns = colnames
    
    mv_cols = [x for x in tweet_df.columns if x.find('_mv') > -1]
    mv_counts_df = tweet_df[mv_cols+['UserID']].groupby('UserID').agg('sum')
    sentiment_dist_df = sentiment_dist_df.merge(mv_counts_df, left_index=True, right_index=True)
    sentiment_dist_df = sentiment_dist_df.merge(tweet_df[['UserID','label']].drop_duplicates(
            ).set_index('UserID'), left_index=True, right_index=True)
    
    # Save transformed to disk
    sentiment_dist_df.to_csv(main_directory + '/data/sentiment_dist.csv')
    
if __name__ == "__main__":
    
    main()
    