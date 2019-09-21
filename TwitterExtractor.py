# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'TweeterClassifier'))
	print(os.getcwd())
except:
	pass

#%%
try:
    import json
except ImportError:
    import simplejson as json

# Import the tweepy library
import tweepy
import csv

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '841391960-0tbN4eZm9J8HG4DNYvr1Pfw6DmFGSDOZU3VnaiwG'
ACCESS_SECRET = 'jQTdTsq7EdrnQx9NkniIoQa0HQSIZRgvTEsnqAO08U9Z4'
CONSUMER_KEY = 'eLgh0mtAHCjC47oQoWHt290oF'
CONSUMER_SECRET = '67CQtjlGlW6HgYVEffNCFHfylsfsp6mX5fcaphHrV08jBku9SE'


#%%
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


#%%
def writeIntoFile(tweets,fileName) :
    
    csv.register_dialect('myDialect',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    
    with open(fileName + '.csv', 'w') as writeFile:
        writer = csv.writer(writeFile,dialect='myDialect')
        writer.writerows(tweets)

    writeFile.close()


#%%
def getTweetFromTwitter(hashTag,classification):
    tweetsToWrite = []
    for status in tweepy.Cursor(api.search,q=hashTag,count=20,
                           lang="en").items():
        tweet = status._json
        if 'text' in tweet: # only messages contains 'text' field is a tweet
            tweetfromHashTag = [str(tweet['id']), tweet['text'].encode("utf-8"),classification]
            tweetsToWrite.append(tweetfromHashTag)
            
    if tweetsToWrite.count != 0:
        writeIntoFile(tweetsToWrite,classification)


#%%
getTweetFromTwitter("#profit -filter:@markets","profit")


#%%



