from twython import Twython
import numpy as np
import pandas as pd
import os


screenNames = {'urbandaddy', 'nyudatascience', 'spacex'}

USERLISTFILE = "TwitterUsers.csv"
CWD = os.getcwd() + os.sep

#replace these with your own app key and secret
APP_KEY = '720258860880015361-eucto9mH525tPCqMFh7ZKHcADWY8Lkx'
APP_SECRET = 'TpI9RJsHiqmUERIq9ISuwyZPjwFBtMfwYgkW09MRvKEIl'

#update this to save it locally and only grab it the first time.
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

def userReputation(user):
    #get friends and followers count
    friends_count = [item['friends_count'] for item in user][0]
    followers_count = [item['followers_count'] for item in user][0]    
    rep = followers_count/(friends_count + followers_count)
    
    return(rep)

def userAccountTaste(sName):
    
    friends_nested = []
    next_cursor = -1
    
    while(next_cursor != 0):
        #Get page of friends
        obj = (twitter.get_friends_list(screen_name=sName, count = 200, skip_status = True, cursor = next_cursor))
        #Append to list
        friends_nested.append(obj['users'])
        #update cursor
        next_cursor = obj['next_cursor']
    
    #flatten list
    friend_list = [item for sublist in friends_nested for item in sublist]
    
    #Get friends and followers lists
    friends_count = [item['friends_count'] for item in friend_list]
    followers_count = [item['followers_count'] for item in friend_list]
    taste = np.mean([fo/(fr + fo) for fr, fo in zip(friends_count, followers_count)])
    
    return(taste)

for sName in screenNames:
    print("Scanning:", sName)

    user  = twitter.lookup_user(screen_name=sName)    
    reputation = userReputation(user)
    account_taste = userAccountTaste(sName)    

    temp = pd.DataFrame({'screen_name':[sName], 
                         'reputation': [reputation], 
                         'account_taste': [account_taste]})
    twitUsers = twitUsers.append(temp)



