#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:51:07 2017

@author: stephencarrow
"""

import pandas as pd
import numpy as np
import collections
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import minmax_scale

class SentimentExtractor:
        
    def TweetsVADER(self, tweets, tweet_ids):
        
        analyzer = SentimentIntensityAnalyzer()
        
        compound = []
        neg = []
        pos = []
        neu = []
        for each_tweet in tweets:
            try:
                scores = analyzer.polarity_scores(each_tweet)
            except:
                scores = {'compound': np.nan, 'neg': np.nan, 'neu': np.nan, 'pos': np.nan}
    
            compound += [scores['compound']]
            neg += [scores['neg']]
            pos += [scores['pos']]
            neu += [scores['neu']]
    
        vader_df = pd.DataFrame({'compound': compound,
                                 'neg': neg,
                                 'neu': neu,
                                 'pos': pos})
            
        vader_df['TweetID'] = tweet_ids
    
        return vader_df
    
    
    def _ScoreVAD(self, sentence, dictionary_df):
        
        tokens = word_tokenize(sentence)
        
        vad_score = collections.defaultdict(list)
        for each_score in ['V', 'A', 'D']:
            
            inv_std = []
            inv_std_sum = 0.0
            a_mu = []
            
            not_exists = 0
            for each_word in tokens:
                if each_word.lower() in dictionary_df.index:
                    each_word = each_word.lower()
                    try:
                        mean = dictionary_df.loc[each_word][each_score+'.Mean.Sum']
                        std = dictionary_df.loc[each_word][each_score+'.SD.Sum']
                    except:
                        pass
                    
                    inv_std_k = 1.0/std
                    inv_std.append(inv_std_k)
                    inv_std_sum += inv_std_k
    
                    a_mu.append(mean)
                    
                else:
                    not_exists += 1
            
            if not_exists < len(tokens):        
                score = 0.0
                for i in range(len(a_mu)):
                    score += inv_std[i]/inv_std_sum*a_mu[i]
            else:
                score = np.nan
                
            vad_score[each_score] = score
    
        return vad_score
    
    
    def TweetsANEW(self, tweets, tweet_ids, dictionary_df):
        valence = []
        arousal = []
        dominance = []
        
        for each_tweet in tweets:
            try:
                scores = self._ScoreVAD(each_tweet, dictionary_df)
            except:
                scores = {'V': np.nan, 'A': np.nan, 'D': np.nan}
                
            valence += [scores['V']]
            arousal += [scores['A']]
            dominance += [scores['D']]
    
        vad_df = pd.DataFrame({'valence': valence,
                               'arousal': arousal,
                               'dominance': dominance})
        
    
        vad_df_scaled = vad_df.apply(lambda x: minmax_scale(x, (-1,1)) if x.dtype!=int else x)
        
        vad_df_scaled['TweetID'] = tweet_ids
        
        return vad_df_scaled