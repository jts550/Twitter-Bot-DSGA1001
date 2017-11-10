from twython import Twython
import pandas as pd
import os
import argparse

#replace these with your own app key and secret
APP_KEY = 'HOShGCrDyAdKRgudjn6wF1Zzg'
APP_SECRET = 'IPCJVbKU5sJd85twNv6qgvRQXp4gFE5133bIprV76lBx3n5Jay'


#Our Argument settings and parsing
parser = argparse.ArgumentParser(description='Load the Twitter account details from a list of ids')
parser.add_argument('file',  type=str, help='a file to load ids from')

parser.add_argument('--no_update', action='store_true', help="skip rather than update existing ids (default is to update)")

parser.add_argument('--label',  default=None, type=int, choices=(0,1),
                   help="label all the ids as Bot(1) or Not(0) (default is take from file")

parser.add_argument('--outfile',  default="TwitterUsers.csv", type=str,
                   help="label all the ids as Bot(1) or Not(0) (default is take from file")

args = parser.parse_args()


#check if we can load ids from the given list
idFile =  args.file
if (os.path.isfile(idFile)):
    print("loading from", idFile)
else :
    exit("ID file not found.")

idList = pd.read_csv(idFile,header=None)
allLabels = -1
if (idList.shape[1] ==1):
    if (args.label == None):
        exit("No labels in file or specified at runtime")
    else :
        allLabels = args.label
        print("Using lablel",allLabels)


#update this to save it locally and only grab it the first time.
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

#establish a twitter connection with our info
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

#initalize our pandas DF (probably not needed in Python?
twitUsers = None

#load any existing users in our output file
ulFile = args.outfile
if (os.path.isfile(ulFile)):
    twitUsers = pd.read_csv(ulFile, index_col='screen_name', encoding = "cp1252")

#check if we want to update in place, or skip existing
updateInPlace = True
if args.no_update:
    updateInPlace = False
    print("Skipping already loaded users.")

countNotFound = 0
countSkipped = 0
countAddedUsers = 0

#Right now we look up each user individually, could change to group in bunches of 100
for tid in idList.iloc[:, 0] :
    stid = str(tid)

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
        #create, update, or append depending on existing state of dataframe
        if (not isinstance(twitUsers, pd.DataFrame)):
            twitUsers = pd.DataFrame(user, index=[user['screen_name']])
            countAddedUsers = countAddedUsers + 1
        elif (user['screen_name'] in twitUsers.index):
            twitUsers.update(pd.DataFrame(user, index=[user['screen_name']]))
        else:
            twitUsers = twitUsers.append(pd.DataFrame(user, index=[user['screen_name']]))
            countAddedUsers = countAddedUsers + 1

    tweetList = None
    '''
    tweets = twitter.get_user_timeline(user_id=stid)
    for tweet in tweets:
        if not printed_tweet:
            print(tweet)
            printed_tweet = True

        if tweetList is None:
            tweetList = pd.DataFrame.from_dict([tweet]).drop('user',1 )
        else:
            tweetList = tweetList.append(pd.DataFrame.from_dict([tweet]).drop('user',1))
    '''
    if (tweetList is not None):
        print("Saving ", tweetList.shape[0], "for", tid)
        #tweetList.to_csv(CWD+stid+"-tweets.csv")
    else:
        print("Done")

print("Saving ", twitUsers.shape[0], "users, added:", countAddedUsers, "skipped:", countSkipped, "couldn't find:", countNotFound )
twitUsers.to_csv(ulFile, index=True, index_label='screen_name', encoding = "cp1252")
