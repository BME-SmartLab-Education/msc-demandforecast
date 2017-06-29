# -*- coding: utf-8 -*-


import time
import pandas as pd
from pandas.tseries.offsets import Day
import os
import got3


#Process tweet blocks
def add_tweets_to_collection(ls_tw, tws):
    for tweet in tws:
        ls_data = [
                   tweet.id,
                   tweet.date.strftime('%Y-%m-%d'),
                   tweet.username,
                   tweet.retweets,
                   tweet.favorites,
                   tweet.mentions,
                   tweet.text.encode('utf-8')
                   ]
        ls_tw.append(ls_data)
    
    
    
#Time interval
date_start = '2014-10-17'
date_end = '2016-03-07'
file_name = 'tweets_{}_{}.h5'.format(date_start, date_end) #like: tweets_2014-10-17_2014-10-18
QUERY = 'ABC'

#If there is previous tweet file
if os.path.exists(file_name): 
    df_prev_tweets = pd.read_hdf(file_name, 'tweets')
    date_prev_max = pd.to_datetime(df_prev_tweets['DATE'].max(), format='%Y-%m-%d')
    date_prev_max -= Day()
    print('The latest date until this was: {}'.format(date_prev_max))

else:
    date_prev_max = date_start    
    print('New dataset is being created, starting with: {}'.format(date_prev_max))
        
print('Should find all related tweets until {}'.format(date_end))


#Create iterable date range
date_range = pd.date_range(date_prev_max, date_end)

#Collection
ls_tweets = []

#Download tweets    
start = time.time()
try:
    for day_actual in date_range:
        day_next = day_actual + Day()
        day_actual = day_actual.strftime('%Y-%m-%d')  
        day_next = day_next.strftime('%Y-%m-%d')  
        
        print('COLLECTING tweets between {} and {}'.format(day_actual, day_next))
        start_actual = time.time()
        tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(QUERY).setSince(day_actual).setUntil(day_next)#.setMaxTweets(100)
        tweets = got3.manager.TweetManager.getTweets(tweetCriteria)
        duration_actual = time.time() - start_actual
    
        
        add_tweets_to_collection(ls_tweets, tweets)
        print('Scraped {} tweets in {} minutes'.format(len(tweets), duration_actual/60))
        
except(KeyboardInterrupt):
    pass

except:
    pass

duration = time.time()-start
print('Run took {} minutes'.format(duration/60))

   

#Create dataframe
df_tweets = pd.DataFrame(ls_tweets, columns=[
                                             'ID',
                                             'DATE',
                                             'USERNAME',
                                             'RETWEETS',
                                             'FAVORITES',
                                             'MENTIONS',
                                             'TWEETS'
                                             ])


#If there is a previous tweet file --> concatenate them
if os.path.exists(file_name):       
    df_tweets = pd.concat([df_prev_tweets, df_tweets])  
    print('The previous file contained {} tweets'.format(df_prev_tweets.shape[0]))

#Display info
print('The new file contains {} tweets'.format(df_tweets.shape[0]))


#Drop duplicates
df_tweets.drop_duplicates('ID', inplace=True)
print('Dupicates are dropped. New tweet-number is: {}'.format(df_tweets.shape[0]))

#Print numbers by dates
print(df_tweets['DATE'].value_counts())

#Save tweets
df_tweets.to_hdf(file_name, 'tweets')
print('File saved to disk with name: {}.'.format(file_name))
    
    
