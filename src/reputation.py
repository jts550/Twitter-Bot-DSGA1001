#load libraries
import tweepy
import numpy as np
import pandas as pd
import os
import json

#login 
APP_KEY = 'HOShGCrDyAdKRgudjn6wF1Zzg'
APP_SECRET = 'IPCJVbKU5sJd85twNv6qgvRQXp4gFE5133bIprV76lBx3n5Jay'

auth = tweepy.AppAuthHandler(APP_KEY, APP_SECRET)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

lastPerc = 0

#Get User Ids
data_path = os.getcwd() + os.sep + "varol-2017-ids.csv"
ids = pd.read_csv(data_path, usecols = [0] )
twitUsers_incomplete = pd.DataFrame(columns = ["id", 'reputation', 'taste'])
twitUsers_incomplete.id = ids


#reputation
def Reputation(user):

    rep = user.followers_count/(user.friends_count + user.followers_count)

    return(rep)

#account taste
def AccountTaste(user):
    friends_rep = []
    friendList = api.friends_ids(user_id = user.id)
    print(" fl:", len(friendList), " ", end='')
    for friend in friendList:
        try:
            user = api.get_user(user_id  = friend) 
            friends_rep.append(Reputation(user))
        except:
           print("died on:", friend)
    taste = np.nanmean(friends_rep)
    #print("taste:", taste);
    return(taste)

def doneYet(account):
    global lastPerc
    perc = np.round(twitUsers_incomplete.loc[twitUsers_incomplete['id']==account.id].index[0] / ids.shape[0], decimals = 2) * 100
    if(perc > lastPerc ):
        print("%d%% Done" %perc)
        lastPerc = perc


def FillAccount(account):
    doneYet(account)
    user=None
    try:
        print("processing:", account.id, end='')
        user = api.get_user(user_id  = account.id)
        account.reputation = Reputation(user)
        account.taste = AccountTaste(user)
        print("finished with:", user.id)
    except tweepy.TweepError as e:
        print("TweepError:",e)
        account.reputation = np.nan
        account.taste = np.nan
        if (user != None):
           print("bailed on :", user.id)

    return(account)

#twitUsers = twitUsers_incomplete.apply(FillAccount, ids, axis=1)
twitUsers = twitUsers_incomplete.apply(FillAccount, axis=1)

csv_path = os.getcwd() + os.sep + "reputation.csv"
twitUsers.to_csv(csv_path, index = False)

