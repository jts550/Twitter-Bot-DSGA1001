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
    
    # Load api dump data
    cwd = os.getcwd()
    main_directory = '/'.join(cwd.split('/')[:-1])

    tweet_df = pd.read_csv(main_directory + '/data/varol_cleaned.csv')
    
    # Load ANEW dictionary
    anew_df = pd.read_csv(main_directory + '/data/Ratings_Warriner_et_al.csv',
                         header=0, index_col=0)
    anew_df.set_index('Word', inplace=True)
    
    #Apply sentiment feature extractors        
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
    sentiment_dist_df.to_csv(main_directory + '/data/sentiment_dist_varol_dump.csv')
    
if __name__ == "__main__":
    
    main()
    