import tweepy #https://github.com/tweepy/tweepy
import csv
import pandas as pd
import re


def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #Twitter API credentials
    consumer_key = ""
    consumer_secret = ""
    access_key = ""
    access_secret = ""
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")
    
    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
    
    #write the csv  
    with open(f'new_{screen_name}_tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
    
    pass

def clean_csv(fname='new_SICKOFWOLVES_tweets.csv'):
    df = pd.read_csv(fname)
    df = df.dropna()
    df = df[~df['text'].str.contains('STORE')]
    df = df[~df['text'].str.startswith('@')]
    df['text'] = df['text'].str.replace('\n', ' ')
    df['text'] = df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]) # remove urls
    df['text'] = df["text"].apply(lambda x:''.join([" " if ord(i) < 32 or ord(i) > 126 # remove non-ascii
                                                    else i for i in x]))
    df['text'].to_csv('titles.csv', index=False, header=False)
