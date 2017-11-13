from twython import Twython
from twython import TwythonError
import pandas as pd
import os
import argparse
import numpy as np
import time
import sys

#replace these with your own app key and secret
APP_KEY = 'HOShGCrDyAdKRgudjn6wF1Zzg'
APP_SECRET = 'IPCJVbKU5sJd85twNv6qgvRQXp4gFE5133bIprV76lBx3n5Jay'

print("Starting up....")
#Our Argument settings and parsing
parser = argparse.ArgumentParser(description='Load the Twitter account details from a list of ids')
parser.add_argument('file',  type=str, help='a file to load ids from')
parser.add_argument('--no_update', action='store_true', help="skip rather than update existing ids (default is to update)")
parser.add_argument('--remove', action='store_true', help="REMOVES entries from our existing output file")
parser.add_argument('--notweets', action='store_true', help="don't scan the tweets, just account data")
parser.add_argument('--label',  default=None, type=int, choices=(0,1),
                   help="label all the ids as Bot(1) or Not(0) (default is take from file")
parser.add_argument('--userfile',  default="TwitterUsers.csv", type=str,
                   help="file to serialize our dataframe to, defaults to TwitterUsers.csv")
parser.add_argument('--tweetfile',  default="TwitterTweets.csv", type=str,
                   help="file to serialize our tweet dataframe to, defaults to TwitterTweets.csv")
parser.add_argument('--pauseFreq',  default="10000", type=int,
                   help="after how many users requests to pause")
parser.add_argument('--pauseLength',  default="0", type=int,
                   help="how long to pause")

args = parser.parse_args()

print("Parsed params")

#check if we can load ids from the given list
idFile =  args.file
if (os.path.isfile(idFile)):
    print("Loading ID list from", idFile)
else :
    exit("ID file not found.")

idList = pd.read_csv(idFile,header=None)

if (idList.shape[1] ==1):
    if (args.label == None):
        exit("No labels in file or specified at runtime")
    else :
        idList['label'] = args.label
        print("Using label",args.label)

idList.columns = ['id', 'label']
idList.set_index('id', inplace=True, drop=False)


#initalize our pandas DF
twitUsers = None

# load any existing users in our output file
userFile = args.userfile
if (os.path.isfile(userFile)):
    twitUsers = pd.read_csv(userFile, encoding="cp1252")
    twitUsers.set_index('id_str', drop=False, inplace=True)

# load any existing tweets in our output file
masterTweetList = None
tweetFile = args.tweetfile
if (os.path.isfile(tweetFile)):
    masterTweetList= pd.read_csv(tweetFile,encoding="cp1252" )
    masterTweetList.set_index('id_str', drop=False, inplace=True)

#some useful counts for output
countNotFound = 0
countSkipped = 0
countAddedUsers = 0
countUpdatedUsers = 0
countAddedTweets = 0
countUpdateTweets = 0
countRemoved = 0

#save the files, this is usign global variables and terrible
def saveFiles():
    if (isinstance(twitUsers, pd.DataFrame)):
        print("Saving ", twitUsers.shape[0], "users, added:", countAddedUsers, "updated:", countUpdatedUsers,
          "skipped:", countSkipped, "couldn't find:", countNotFound, "REMOVED:", countRemoved)
        twitUsers.to_csv(userFile, index=False)  # , encoding = "cp1252")
    if (isinstance(masterTweetList, pd.DataFrame)):
        print("Saving ", masterTweetList.shape[0], "tweets")
        masterTweetList.to_csv(tweetFile, index=False)  # , encoding = "cp1252")


# Check if we just want to remove, and do so
if (args.remove):
    print("Removing ids from", userFile)
    if (not isinstance(twitUsers, pd.DataFrame)) :
        print("No data to remove from")
        exit()

    countRemoved = 0
    for tid in idList.iloc[:, 0]:
        if (tid in twitUsers['id_str'].values):
            twitUsers.drop(twitUsers[twitUsers.id_str == tid].index, inplace=True)
            if (isinstance(masterTweetList, pd.DataFrame)):
                masterTweetList.drop(masterTweetList[masterTweetList.id_str == tid].index, inplace=True)
            countRemoved = countRemoved + 1

    saveFiles()
    exit()

# check if we want to update in place, or skip existing
updateInPlace = True
if args.no_update:
    updateInPlace = False
    print("Skipping already loaded users.")

#check if we want tweets
saveTweets = True
if args.notweets:
    saveTweets = False
    print("Skipping tweets.")

sys.stdout.flush()

#update this to save it locally and only grab it the first time.
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

#establish a twitter connection with our info
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

#for pausing x seconds every y records
requestCounter = 0

#Right now we look up each user individually, could change to group in bunches of 100
for tid in idList.iloc[:, 0] :
    stid = str(tid)
    label = idList.loc[tid]['label']

    if (isinstance(twitUsers, pd.DataFrame) and tid in twitUsers['id_str'].index and not updateInPlace):
        countSkipped = countSkipped  + 1
        print("Skipping:", tid, "- ", end='')
        results = {}
    else :
        print("Scanning:", tid,"- ", end='')
        try:
            requestCounter = requestCounter + 1
            results  = twitter.lookup_user(user_id=stid, entities=False)
        except TwythonError as e:
            results = {}
            print("User not found:", e.error_code, "- ", end='')
            countNotFound = countNotFound + 1

    for user in results :
        print(user['screen_name']," - ", end='')
        user['label'] = label

        #create, update, or append depending on existing state of dataframe
        if (not isinstance(twitUsers, pd.DataFrame)):
            twitUsers = pd.DataFrame(user, index=[user['id_str']])
            countAddedUsers = countAddedUsers + 1
        elif (np.int64(user['id_str']) in twitUsers.index):
            twitUsers.update(pd.DataFrame(user, index=[user['id_str']]))
            countUpdatedUsers = countUpdatedUsers + 1
        else:
            newdf = pd.DataFrame(user, index=[user['id_str']])
            twitUsers = twitUsers.append(newdf, ignore_index=True)
            #twitUsers = twitUsers.append(pd.DataFrame(user))
            countAddedUsers = countAddedUsers + 1

    #IF we want to save tweets, and we haven't already if no_update is set
    #note that if we failed to find the user, we still try and load a status, using up some of our rate limit
    #this could be improved
    if (saveTweets and not (isinstance(masterTweetList, pd.DataFrame) and tid in masterTweetList['id_str'].index and not updateInPlace)):
        try:
            requestCounter = requestCounter + 1
            tweets = twitter.get_user_timeline(user_id=stid, count=200, exclude_replies=True)
        except TwythonError as e:
            tweets = {}
            print("Can't load tweets (private?):", e.error_code)
            continue
    else:
        tweets = {}

    countAddedTweets = 0
    countUpdateTweets = 0

    for tweet in tweets:
        newTweet = pd.DataFrame.from_dict([tweet])
        newTweet['label'] = label
        newTweet['user_id'] = tid
        newTweet.set_index('id_str', inplace=True, drop=False)
        newTweet.drop('entities',1, inplace=True)
        newTweet.drop('user', 1, inplace=True)
        if ('extended_entities' in newTweet.columns):
            newTweet.drop('extended_entities', 1, inplace=True)
        if ('retweeted_status' in newTweet.columns):
            newTweet.drop('retweeted_status',1, inplace=True)


        # create, update, or append depending on existing state of dataframe
        if (not isinstance(masterTweetList, pd.DataFrame)):
            masterTweetList = pd.DataFrame(newTweet)
            countAddedTweets = countAddedTweets + 1
        elif (np.int64(tweet['id_str']) in masterTweetList.index):
            masterTweetList.update(newTweet)
            countUpdateTweets = countUpdateTweets + 1
        else:
            masterTweetList = masterTweetList.append(newTweet)
            countAddedTweets = countAddedTweets + 1

    if tweets:
        print("Added tweets:", countAddedTweets, "updated tweets:", countUpdateTweets, "master grew to:", masterTweetList.shape[0])
    else:
        print("No tweets found")

    # Pause after so many requests.  Don't count skips, but do count no founds.
    if (requestCounter > args.pauseFreq):
        saveFiles()
        sys.stdout.flush()
        print("Waiting", args.pauseLength, "seconds...")
        sys.stdout.flush()
        time.sleep(args.pauseLength)
        requestCounter = 0
    sys.stdout.flush()

saveFiles()
