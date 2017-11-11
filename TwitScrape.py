from twython import Twython
import pandas as pd
import os
import argparse
import numpy as np

#replace these with your own app key and secret
APP_KEY = 'HOShGCrDyAdKRgudjn6wF1Zzg'
APP_SECRET = 'IPCJVbKU5sJd85twNv6qgvRQXp4gFE5133bIprV76lBx3n5Jay'


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
args = parser.parse_args()

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
    twitUsers.set_index('screen_name', drop=False, inplace=True)

# load any existing tweets in our output file
masterTweetList = None
tweetFile = args.tweetfile
if (os.path.isfile(tweetFile)):
    masterTweetList= pd.read_csv(tweetFile,encoding="cp1252" )
    masterTweetList.set_index('id_str', drop=False, inplace=True)


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
            countRemoved = countRemoved + 1

    print("Saving ", twitUsers.shape[0], "users, removed:", countRemoved)
    twitUsers.to_csv(userFile, index=True, index_label='screen_name', encoding = "cp1252")
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

#update this to save it locally and only grab it the first time.
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

#establish a twitter connection with our info
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)


#some useful counts for output
countNotFound = 0
countSkipped = 0
countAddedUsers = 0
countUpdatedUsers = 0

#Master list of tweets
#masterTweetList = None

#Right now we look up each user individually, could change to group in bunches of 100
for tid in idList.iloc[:, 0] :
    stid = str(tid)
    label = idList.loc[tid]['label']

    if (isinstance(twitUsers, pd.DataFrame) and tid in twitUsers['id_str'].values and not updateInPlace):
        countSkipped = countSkipped  + 1
        print("skipping:",stid)
        continue

    print("Scanning:", tid,":", end='')
    try:
        results  = twitter.lookup_user(user_id=stid, entities=False)
    except:
        results = {}
        print("User not found")
        continue

    for user in results :
        user['label'] = label
        #create, update, or append depending on existing state of dataframe
        #print("looking for",user['screen_name'], "in",twitUsers.index  )

        if (not isinstance(twitUsers, pd.DataFrame)):
            twitUsers = pd.DataFrame(user, index=[user['screen_name']])
            countAddedUsers = countAddedUsers + 1
        elif (user['screen_name'] in twitUsers.index):
            twitUsers.update(pd.DataFrame(user, index=[user['screen_name']]))
            countUpdatedUsers = countUpdatedUsers + 1
        else:
            twitUsers = twitUsers.append(pd.DataFrame(user, index=[user['screen_name']]))
            countAddedUsers = countAddedUsers + 1

    if saveTweets:
        tweets = twitter.get_user_timeline(user_id=stid, count=200, exclude_replies=True)
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

    print("Added tweets:", countAddedTweets, "updated tweets:", countUpdateTweets, "master grew to:", masterTweetList.shape[0])


print("Saving ", twitUsers.shape[0], "users, added:", countAddedUsers, "updated:", countUpdatedUsers, "skipped:", countSkipped, "couldn't find:", countNotFound )
twitUsers.to_csv(userFile, index=False, encoding = "cp1252")
print("Saving ", masterTweetList.shape[0], "tweets")
masterTweetList.to_csv(tweetFile, index=False)#, encoding = "cp1252")